import os

import numpy as np
import pytest

from cell_eval import MetricsEvaluator
from cell_eval.data import (
    CELLTYPE_COL,
    CONTROL_VAR,
    N_CELLTYPES,
    PERT_COL,
    build_random_anndata,
    downsample_cells,
)

OUTDIR = "TEST_OUTPUT_DIRECTORY"

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


def test_unknown_alternative_de_metric():
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
            metric="unknown",
        ).compute()


def test_eval_simple():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
        celltype_col="celltype",
    )
    evaluator.compute()


def test_eval_missing_celltype_col():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)

    adata_real.obs.drop(columns="celltype", inplace=True)
    adata_pred.obs.drop(columns="celltype", inplace=True)

    assert "celltype" not in adata_real.obs.columns
    assert "celltype" not in adata_pred.obs.columns

    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
    )
    evaluator.compute()


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


def test_eval_alt_metric():
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
        metric="anderson",
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


def test_eval_alt_fdr_threshold():
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
        fdr_threshold=0.01,
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
