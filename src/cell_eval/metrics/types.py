"""Types module for metric computation."""

import enum
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, TypeVar

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class DESortBy(enum.Enum):
    """Sorting options for differential expression results."""

    FOLD_CHANGE = "fold_change"
    ABS_FOLD_CHANGE = "abs_fold_change"
    PVALUE = "p_value"
    FDR = "fdr"


@dataclass(frozen=True)
class DEResults:
    """Raw differential expression results with sorting and filtering capabilities."""

    data: pd.DataFrame
    control_pert: str

    # Column names configuration
    target_col: str = "target"
    feature_col: str = "feature"
    fold_change_col: str = "fold_change"
    pvalue_col: str = "p_value"
    fdr_col: str = "fdr"

    def __post_init__(self) -> None:
        required_cols = {
            self.target_col,
            self.feature_col,
            self.fold_change_col,
            self.pvalue_col,
            self.fdr_col,
        }
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure numeric columns are float32
        object.__setattr__(
            self,
            "data",
            self.data.assign(
                **{
                    self.fold_change_col: self.data[self.fold_change_col].astype(
                        "float32"
                    ),
                    self.pvalue_col: self.data[self.pvalue_col].astype("float32"),
                    self.fdr_col: self.data[self.fdr_col].astype("float32"),
                }
            ),
        )

    def get_significant_genes(
        self, pert: str, fdr_threshold: float = 0.05
    ) -> pd.Series:
        """Get significant genes for a perturbation."""
        return self.data[
            (self.data[self.target_col] == pert)
            & (self.data[self.fdr_col] < fdr_threshold)
        ][self.feature_col]

    def filter_control(self) -> pd.DataFrame:
        """Return data without control perturbation rows."""
        return self.data[self.data[self.target_col] != self.control_pert]

    def sort_by_metric(
        self,
        metric: DESortBy,
        ascending: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Sort DE results by specified metric."""
        if ascending is None:
            ascending = metric in {DESortBy.PVALUE, DESortBy.FDR}

        df = self.filter_control()

        # Add abs_fold_change if needed
        if metric == DESortBy.ABS_FOLD_CHANGE:
            df = df.assign(abs_fold_change=df[self.fold_change_col].abs())

        return df.sort_values(
            [self.target_col, metric.value], ascending=[True, ascending]
        )

    def get_top_genes(
        self,
        sort_by: DESortBy,
        fdr_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Get top genes per perturbation, optionally filtered by FDR."""
        df = self.filter_control()

        # Apply FDR filter if specified
        if fdr_threshold is not None:
            df = df[df[self.fdr_col] < fdr_threshold]

        # Sort by metric
        df = self.sort_by_metric(sort_by)

        # Add rank and pivot
        df = df.assign(rank=df.groupby(self.target_col).cumcount())
        return df.pivot(
            index=self.target_col, columns="rank", values=self.feature_col
        ).sort_index(axis=1)


@dataclass
class DEComparison:
    """Comparison between real and predicted DE results."""

    real: DEResults
    pred: DEResults

    def __post_init__(self) -> None:
        if self.real.control_pert != self.pred.control_pert:
            raise ValueError("Control perturbations don't match")

        real_perts = np.unique(self.real.data[self.real.target_col])
        pred_perts = np.unique(self.pred.data[self.pred.target_col])
        if not np.array_equal(real_perts, pred_perts):
            raise ValueError(
                f"Perturbation mismatch: real {real_perts} != pred {pred_perts}"
            )
        if self.real.control_pert in real_perts:
            raise ValueError(
                f"Control perturbation unexpected in {self.real.control_pert} found in real data: {real_perts}"
            )
        if self.pred.control_pert in pred_perts:
            raise ValueError(
                f"Control perturbation unexpected in {self.pred.control_pert} found in pred data: {pred_perts}"
            )
        object.__setattr__(self, "perturbations", list(real_perts))
        object.__setattr__(self, "n_perts", len(real_perts))

    def iter_perturbations(self) -> Iterator[str]:
        for pert in self.perturbations:
            yield pert

    def compute_overlap(
        self,
        k: Optional[int] = None,
        topk: Optional[int] = None,
        fdr_threshold: Optional[float] = None,
        sort_by: DESortBy = DESortBy.ABS_FOLD_CHANGE,
    ) -> Dict[str, float]:
        """
        Compute overlap metrics across perturbations.

        Args:
            k: If specified, use top k genes from real data
            topk: If specified, use top k genes from predicted data
            fdr_threshold: If specified, only consider genes below this FDR
            sort_by: Metric to sort genes by

        Returns:
            Dictionary mapping perturbation names to overlap scores
        """
        if k is not None and topk is not None:
            raise ValueError("Provide only one of k or topk")

        overlaps = {}
        for pert in self.iter_perturbations():
            # Get sorted gene lists
            real_genes = (
                self.real.get_top_genes(sort_by=sort_by, fdr_threshold=fdr_threshold)
                .loc[pert]
                .dropna()
            )

            pred_genes = (
                self.pred.get_top_genes(sort_by=sort_by, fdr_threshold=fdr_threshold)
                .loc[pert]
                .dropna()
            )

            # Apply k/topk limits
            if k == -1:
                k_eff = len(real_genes)
            else:
                k_eff = k if k is not None else len(real_genes)

            real_subset = set(real_genes[:k_eff])
            pred_subset = set(pred_genes[: (topk or k_eff)])

            # Calculate overlap
            denom = len(pred_subset) if topk else len(real_subset)
            overlaps[pert] = len(real_subset & pred_subset) / denom if denom else 0.0

        return overlaps


Array = TypeVar("Array", bound=NDArray)


@dataclass(frozen=True)
class PerturbationAnndataPair:
    """Pair of AnnData objects with perturbation information."""

    real: ad.AnnData
    pred: ad.AnnData
    pert_col: str
    control_pert: str

    def __post_init__(self) -> None:
        if self.real.shape[1] != self.pred.shape[1]:
            raise ValueError(
                f"Shape mismatch: real {self.real.shape[1]} != pred {self.pred.shape[1]}"
                " Expected to be the same number of genes"
            )

        perts_real = np.unique(self.real.obs[self.pert_col])
        perts_pred = np.unique(self.pred.obs[self.pert_col])
        if not np.array_equal(perts_real, perts_pred):
            raise ValueError(
                f"Perturbation mismatch: real {perts_real} != pred {perts_pred}"
            )
        perts = np.union1d(perts_real, perts_pred)
        object.__setattr__(self, "perts", perts)

    def build_delta_array(self, pert: str) -> "DeltaArrays":
        """Build delta array for a perturbation."""
        pert_real = self.real.X[self.real.obs[self.pert_col] == pert, :]
        pert_pred = self.pred.X[self.pred.obs[self.pert_col] == pert, :]
        ctrl_real = self.real.X[self.real.obs[self.pert_col] == self.control_pert, :]
        ctrl_pred = self.pred.X[self.pred.obs[self.pert_col] == self.control_pert, :]
        return DeltaArrays(
            pert=pert,
            pert_real=pert_real,
            pert_pred=pert_pred,
            ctrl_real=ctrl_real,
            ctrl_pred=ctrl_pred,
        )

    def iter_delta_arrays(self) -> Iterator["DeltaArrays"]:
        """Iterate over delta arrays for all perturbations."""
        for pert in self.perts:
            if pert == self.control_pert:
                continue
            yield self.build_delta_array(pert)


@dataclass(frozen=True)
class DeltaArrays:
    """Arrays for computing differences from control."""

    pert: str
    pert_real: Array
    pert_pred: Array
    ctrl_real: Array
    ctrl_pred: Optional[Array] = None

    def __post_init__(self) -> None:
        # Validate shapes match (only number of genes)
        shapes = {
            "pert_real": self.pert_real.shape[1],
            "pert_pred": self.pert_pred.shape[1],
            "ctrl_real": self.ctrl_real.shape[1],
        }
        if self.ctrl_pred is not None:
            shapes["ctrl_pred"] = self.ctrl_pred.shape[1]

        if len(set(shapes.values())) > 1:
            raise ValueError(f"Shape mismatch in arrays: {shapes}")
