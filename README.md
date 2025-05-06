# state-eval

Set config parameters in `eval_config.yaml`

## Description

This package provides evaluation metrics for single-cell perturbation predictions.

Differential expression analysis (parallel wilcoxon rank sum) is performed using [`pdex`](https://github.com/arcinstitute/pdex).

## Installation

Distribution with [`uv`](https://docs.astral.sh/uv/)

```bash
# install from github directly
uv pip install git+ssh://github.com/arcinstitute/state-eval

# install from source
git clone ssh://github.com/arcinstitute/state-eval
cd state-eval
uv pip install -e .
```

## Usage

```bash
# run evaluation
uv run run_eval \
    --adata_pred <your/path/to/pred>.h5ad \
    --adata_true <your/path/to/true>.h5ad
```
