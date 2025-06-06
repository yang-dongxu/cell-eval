from dataclasses import dataclass, field
from typing import Iterator, Literal

import anndata as ad
import numpy as np


@dataclass(frozen=True)
class PerturbationAnndataPair:
    """Pair of AnnData objects with perturbation information."""

    real: ad.AnnData
    pred: ad.AnnData
    pert_col: str
    control_pert: str
    perts: np.ndarray[str] = field(init=False)

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
        perts = np.array([p for p in perts if p != self.control_pert])
        object.__setattr__(self, "perts", perts)

    def get_perts(self, include_control: bool = False) -> np.ndarray[str]:
        """Get all perturbations."""
        if include_control:
            return self.perts
        return self.perts[self.perts != self.control_pert]

    def build_delta_array(
        self, pert: str, embed_key: str | None = None
    ) -> "DeltaArrays":
        """Build delta array for a perturbation."""
        mask_pert_real = self.real.obs[self.pert_col] == pert
        mask_pert_pred = self.pred.obs[self.pert_col] == pert
        mask_ctrl_real = self.real.obs[self.pert_col] == self.control_pert
        mask_ctrl_pred = self.pred.obs[self.pert_col] == self.control_pert

        if not embed_key:
            pert_real = self.real.X[mask_pert_real, :]
            pert_pred = self.pred.X[mask_pert_pred, :]
            ctrl_real = self.real.X[mask_ctrl_real, :]
            ctrl_pred = self.pred.X[mask_ctrl_pred, :]
        else:
            if embed_key not in self.real.obsm:
                raise ValueError(f"Embed key {embed_key} not found in real AnnData")
            if embed_key not in self.pred.obsm:
                raise ValueError(f"Embed key {embed_key} not found in pred AnnData")
            pert_real = self.real.obsm[embed_key][mask_pert_real, :]
            pert_pred = self.pred.obsm[embed_key][mask_pert_pred, :]
            ctrl_real = self.real.obsm[embed_key][mask_ctrl_real, :]
            ctrl_pred = self.pred.obsm[embed_key][mask_ctrl_pred, :]

        return DeltaArrays(
            pert=pert,
            pert_real=pert_real,
            pert_pred=pert_pred,
            ctrl_real=ctrl_real,
            ctrl_pred=ctrl_pred,
            embed_key=embed_key,
        )

    def iter_delta_arrays(
        self, embed_key: str | None = None
    ) -> Iterator["DeltaArrays"]:
        """Iterate over delta arrays for all perturbations."""
        for pert in self.perts:
            yield self.build_delta_array(pert, embed_key=embed_key)


@dataclass(frozen=True)
class DeltaArrays:
    """Arrays for computing differences from control."""

    pert: str
    pert_real: np.ndarray
    pert_pred: np.ndarray
    ctrl_real: np.ndarray
    ctrl_pred: np.ndarray | None = None
    embed_key: str | None = None

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

    def perturbation_effect(
        self, which: Literal["real", "pred"] = "real", abs: bool = False
    ) -> np.ndarray:
        match which:
            case "real":
                effect = self.pert_real.mean(axis=0) - self.ctrl_real.mean(axis=0)
            case "pred":
                if self.ctrl_pred is None:
                    effect = self.pert_pred.mean(axis=0) - self.ctrl_real.mean(axis=0)
                else:
                    effect = self.pert_pred.mean(axis=0) - self.ctrl_pred.mean(axis=0)
            case _:
                raise ValueError(f"Invalid which: {which}")
        if abs:
            effect = np.abs(effect)
        return effect
