# vc-eval

Set config parameters in `eval_config.yaml`

## Description

This package provides evaluation metrics for single-cell perturbation predictions.

Differential expression analysis (parallel wilcoxon rank sum) is performed using [`pdex`](https://github.com/arcinstitute/pdex).

## Installation

Distribution with [`uv`](https://docs.astral.sh/uv/)

```bash
# install from github directly
uv pip install git+ssh://github.com/arcinstitute/vc-eval

# install from source
git clone ssh://github.com/arcinstitute/vc-eval
cd vc-eval
uv pip install -e .
```

## Usage

### CLI Usage

You can run evaluation between two anndatas on the CLI

```bash
# run evaluation
uv run run_eval \
    --adata_pred <your/path/to/pred>.h5ad \
    --adata_true <your/path/to/true>.h5ad
```

### Module Usage

You can access evaluation programmatically using the `vc_eval` module.

```python
from vc_eval import MetricsEvaluator
from vc_eval.data import build_random_anndata, downsample_cells

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
```
