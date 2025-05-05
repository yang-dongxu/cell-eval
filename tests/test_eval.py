import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from state_eval import MetricsEvaluator

PERT_COL = "perturbation"
CELLTYPE_COL = "celltype"
CONTROL_VAR = "control"

N_CELLS = 1000
N_GENES = 100
N_PERTS = 10
N_CELLTYPES = 3
MAX_UMI = 1e6

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
) -> ad.AnnData:
    """Sample a random AnnData object."""
    if random_state is not None:
        np.random.seed(random_state)
    return ad.AnnData(
        X=np.random.randint(0, MAX_UMI, size=(n_cells, n_genes)),
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
        assert os.path.exists(
            f"{OUTDIR}/celltype_{x}_downstream_de_results.csv"
        ), f"Expected file for downstream DE results missing for celltype: {x}"
        assert os.path.exists(
            f"{OUTDIR}/celltype_{x}_pred_de_results_control.csv"
        ), f"Expected file for predicted DE results missing for celltype: {x}"
        assert os.path.exists(
            f"{OUTDIR}/celltype_{x}_real_de_results_control.csv"
        ), f"Expected file for real DE results missing for celltype: {x}"


@pytest.mark.xfail
def test_broken_adata_missing_pertcol_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_real
    adata_real.obs.drop(columns=[PERT_COL], inplace=True)

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


@pytest.mark.xfail
def test_broken_adata_missing_pertcol_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_pred
    adata_pred.obs.drop(columns=[PERT_COL], inplace=True)

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


@pytest.mark.xfail
def test_broken_adata_missing_celltypecol_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove celltype_col from adata_real
    adata_real.obs.drop(columns=[CELLTYPE_COL], inplace=True)

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


@pytest.mark.xfail
def test_broken_adata_missing_celltypecol_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove celltype_col from adata_pred
    adata_pred.obs.drop(columns=[CELLTYPE_COL], inplace=True)

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


@pytest.mark.xfail
def test_broken_adata_missing_control_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_real
    adata_real = adata_real[adata_real.obs[PERT_COL] != CONTROL_VAR].copy()

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


@pytest.mark.xfail
def test_broken_adata_missing_control_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_pred
    adata_pred = adata_pred[adata_pred.obs[PERT_COL] != CONTROL_VAR].copy()

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
