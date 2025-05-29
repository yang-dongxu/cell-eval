# cell-eval

Set config parameters in `eval_config.yaml`

## Description

This package provides evaluation metrics for single-cell perturbation predictions.

Differential expression analysis (parallel wilcoxon rank sum) is performed using [`pdex`](https://github.com/arcinstitute/pdex).

## Installation

Distribution with [`uv`](https://docs.astral.sh/uv/)

```bash
# install from github directly
uv pip install git+ssh://github.com/arcinstitute/cell-eval

# install from source
git clone ssh://github.com/arcinstitute/cell-eval
cd cell-eval
uv pip install -e .

# install cli with uv tool
uv tool install git+ssh://github.com/arcinstitute/cell-eval
cell-eval --help
```

## Usage

### CLI Usage

You can run evaluation between two anndatas on the CLI

```bash
# prepare for processing / strip anndata to bare essentials + compression
cell-eval prep -i <your/path/to/pred>.h5ad
cell-eval prep -i <your/path/to/real>.h5ad

# run evaluation
cell-eval run \
    -p <your/path/to/pred>.h5ad \
    -r <your/path/to/real>.h5ad
```

### Module Usage

You can access evaluation programmatically using the `cell_eval` module.

```python
from cell_eval import MetricsEvaluator
from cell_eval.data import build_random_anndata, downsample_cells

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
