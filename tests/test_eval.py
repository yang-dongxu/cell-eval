import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from state_eval import MetricsEvaluator

PERT_COL = "perturbation"
CELLTYPE_COL = "celltype"
CONTROL_VAR = "control"

N_CELLS = 1000
N_GENES = 100
N_PERTS = 10
N_CELLTYPES = 3
MAX_UMI = 1e6
NORM_TOTAL = 1e4

RANDOM_SEED = 42

OUTDIR = "TEST_OUTPUT_DIRECTORY"


def build_random_anndata(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    n_perts: int = N_PERTS,
    n_celltypes: int = N_CELLTYPES,
    pert_col: str = PERT_COL,
    celltype_col: str = CELLTYPE_COL,
    control_var: str = CONTROL_VAR,
    random_state: int = RANDOM_SEED,
    as_sparse: bool = False,
    normlog: bool = True,
    normtotal: int = NORM_TOTAL,
) -> ad.AnnData:
    """Sample a random AnnData object."""
    if random_state is not None:
        np.random.seed(random_state)

    # Randomly sample a matrix
    matrix = np.random.randint(0, MAX_UMI, size=(n_cells, n_genes))

    # Normalize and log transform if required
    if normlog:
        matrix = int(normlog) * (matrix / matrix.sum(axis=1).reshape(-1, 1))
        matrix = np.log1p(matrix)

    # Convert to sparse if required
    if as_sparse:
        matrix = csr_matrix(matrix)

    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {
                pert_col: np.random.choice(
                    [f"pert_{i}" for i in range(n_perts)] + [control_var],
                    size=n_cells,
                    replace=True,
                ),
                celltype_col: np.random.choice(
                    [f"celltype_{i}" for i in range(n_celltypes)],
                    size=n_cells,
                    replace=True,
                ),
            }
        ),
    )


def downsample_cells(
    adata: ad.AnnData,
    fraction: float = 0.5,
) -> ad.AnnData:
    """Downsample cells in an AnnData object.

    Copies the output to avoid memory overlaps.
    """
    assert 0 <= fraction <= 1, "Fraction must be between 0 and 1"
    mask = np.random.rand(adata.shape[0]) < fraction
    return adata[mask, :].copy()

def test_missing_adata_input_vars():
    adata_real = build_random_anndata(normlog=False)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=None,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_broken_adata_mismatched_var_size():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()

    # Randomly subset genes on pred
    var_mask = np.random.random(adata_real.shape[1]) < 0.8
    adata_pred = adata_pred[:, var_mask]

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_broken_adata_mismatched_var_ordering():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()

    # Randomly subset genes on pred
    indices = np.arange(adata_real.shape[1])
    np.random.shuffle(indices)
    adata_pred = adata_pred[:, indices]

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_broken_adata_not_normlog():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()

    with pytest.raises(Exception):
        evaluator = MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )
        evaluator.compute()


def test_broken_adata_not_normlog_skip_check():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()
    MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        include_dist_metrics=True,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        celltype_col=CELLTYPE_COL,
        output_space="gene",
        shared_perts=None,
        outdir=OUTDIR,
        class_score=True,
        skip_normlog_check=True,
    )


def test_broken_adata_missing_pertcol_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_real
    adata_real.obs.drop(columns=[PERT_COL], inplace=True)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_broken_adata_missing_pertcol_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_pred
    adata_pred.obs.drop(columns=[PERT_COL], inplace=True)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_broken_adata_missing_celltypecol_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove celltype_col from adata_real
    adata_real.obs.drop(columns=[CELLTYPE_COL], inplace=True)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_broken_adata_missing_celltypecol_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove celltype_col from adata_pred
    adata_pred.obs.drop(columns=[CELLTYPE_COL], inplace=True)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_broken_adata_missing_control_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_real
    adata_real = adata_real[adata_real.obs[PERT_COL] != CONTROL_VAR].copy()

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_broken_adata_missing_control_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_pred
    adata_pred = adata_pred[adata_pred.obs[PERT_COL] != CONTROL_VAR].copy()

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            include_dist_metrics=True,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            celltype_col=CELLTYPE_COL,
            output_space="gene",
            shared_perts=None,
            outdir=OUTDIR,
            class_score=True,
        )


def test_eval():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        include_dist_metrics=True,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        celltype_col=CELLTYPE_COL,
        output_space="gene",
        shared_perts=None,
        outdir=OUTDIR,
        class_score=True,
    )
    evaluator.compute()

    for x in np.arange(N_CELLTYPES):
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_downstream_de_results.csv"), (
            f"Expected file for downstream DE results missing for celltype: {x}"
        )
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_pred_de_results_control.csv"), (
            f"Expected file for predicted DE results missing for celltype: {x}"
        )
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_real_de_results_control.csv"), (
            f"Expected file for real DE results missing for celltype: {x}"
        )


def test_minimal_eval():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        include_dist_metrics=True,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        celltype_col=CELLTYPE_COL,
        output_space="gene",
        shared_perts=None,
        outdir=OUTDIR,
        class_score=True,
        minimal_eval=True,
    )
    evaluator.compute()

    for x in np.arange(N_CELLTYPES):
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_downstream_de_results.csv"), (
            f"Expected file for downstream DE results missing for celltype: {x}"
        )
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_pred_de_results_control.csv"), (
            f"Expected file for predicted DE results missing for celltype: {x}"
        )
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_real_de_results_control.csv"), (
            f"Expected file for real DE results missing for celltype: {x}"
        )


def test_eval_sparse():
    adata_real = build_random_anndata(as_sparse=True)
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        include_dist_metrics=True,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        celltype_col=CELLTYPE_COL,
        output_space="gene",
        shared_perts=None,
        outdir=OUTDIR,
        class_score=True,
    )
    evaluator.compute()

    for x in np.arange(N_CELLTYPES):
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_downstream_de_results.csv"), (
            f"Expected file for downstream DE results missing for celltype: {x}"
        )
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_pred_de_results_control.csv"), (
            f"Expected file for predicted DE results missing for celltype: {x}"
        )
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_real_de_results_control.csv"), (
            f"Expected file for real DE results missing for celltype: {x}"
        )


def test_eval_downsampled_cells():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        include_dist_metrics=True,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        celltype_col=CELLTYPE_COL,
        output_space="gene",
        shared_perts=None,
        outdir=OUTDIR,
        class_score=True,
    )
    evaluator.compute()

    for x in np.arange(N_CELLTYPES):
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_downstream_de_results.csv"), (
            f"Expected file for downstream DE results missing for celltype: {x}"
        )
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_pred_de_results_control.csv"), (
            f"Expected file for predicted DE results missing for celltype: {x}"
        )
        assert os.path.exists(f"{OUTDIR}/celltype_{x}_real_de_results_control.csv"), (
            f"Expected file for real DE results missing for celltype: {x}"
        )
