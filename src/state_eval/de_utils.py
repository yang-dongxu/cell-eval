import logging
import multiprocessing as mp
import os
import time
from collections.abc import Iterator
from functools import partial
from multiprocessing.shared_memory import SharedMemory

import anndata as ad
import numpy as np
import pandas as pd
from adjustpy import adjust
from scipy.stats import ranksums
from tqdm import tqdm

# Configure logger
tools_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PASTED FROM ARC SEQ #


def parallel_compute_de(
    adata_gene,
    control_pert,
    pert_col,
    outdir=None,
    split="real",
    prefix: str = "",
    n_threads: int = 1,
    batch_size: int = 1000,
):
    """
    Compute differential expression using parallel_differential_expression,
    returns two DataFrames: one sorted by fold change and one by p-value

    Parameters
    ----------
    adata_gene : AnnData
        The annotated data matrix with gene expression data
    control_pert : str
        Name of the control perturbation to use as reference
    pert_col : str
        Column in adata_gene.obs that contains perturbation information
    k : int
        Number of top genes to return for each perturbation

    Returns
    -------
    tuple of pd.DataFrame
        Two DataFrames with rows as perturbations and columns as top K genes,
        one sorted by fold change and one by p-value
    """

    # Start timer
    start_time = time.time()

    # Filter groups to only include those with more than 1 cell
    group_counts = adata_gene.obs[pert_col].value_counts()
    valid_groups = group_counts[group_counts > 1].index.tolist()
    adata_gene = adata_gene[adata_gene.obs[pert_col].isin(valid_groups)]

    # Make sure the control perturbation is included in the valid groups
    if control_pert not in valid_groups:
        raise ValueError(
            f"Control perturbation '{control_pert}' has fewer than 2 cells"
        )

    # Run parallel differential expression
    de_results = parallel_differential_expression(
        adata=adata_gene,
        groups=valid_groups,
        reference=control_pert,
        groupby_key=pert_col,
        num_workers=n_threads,
        batch_size=batch_size,
    )

    # # Save out the de results
    if outdir is not None:
        filename = f"{split}_de_results_{control_pert}.csv"
        if prefix:
            filename = f"{prefix}_{filename}"
        outfile = os.path.join(
            outdir,
            filename,
        )
        # if it doesn't already exist, write it out
        if not os.path.exists(outfile):
            de_results.to_csv(outfile, index=False)
        logger.info(f"Saved DE results to {outfile}")
    # #

    logger.info(
        f"Time taken for parallel_differential_expression: {time.time() - start_time:.2f}s"
    )

    # Get DE genes sorted by fold change
    de_genes_fc = vectorized_de(de_results, control_pert, sort_by="abs_fold_change")

    # Get DE genes sorted by p-value
    de_genes_pval = vectorized_de(de_results, control_pert, sort_by="p_value")

    de_genes_pval_fc = vectorized_sig_genes_fc_sort(
        de_results, control_pert, pvalue_threshold=0.05
    )

    de_genes_sig = vectorized_sig_genes_fc_sort(
        de_results, control_pert, pvalue_threshold=0.05
    )

    return de_genes_fc, de_genes_pval, de_genes_pval_fc, de_genes_sig, de_results


def _build_shared_matrix(
    data: np.ndarray,
) -> tuple[str, tuple[int, int], np.dtype]:
    """Create a shared memory matrix from a numpy array."""
    shared_matrix = SharedMemory(create=True, size=data.nbytes)
    matrix = np.ndarray(data.shape, dtype=data.dtype, buffer=shared_matrix.buf)
    matrix[:] = data
    return shared_matrix.name, data.shape, data.dtype


def _conclude_shared_memory(name: str):
    """Close and unlink a shared memory."""
    shm = SharedMemory(name=name)
    shm.close()
    shm.unlink()


def _combinations_generator(
    target_masks: dict[str, np.ndarray],
    var_indices: dict[str, int],
    reference: str,
    target_list: list[str],
    feature_list: list[str],
) -> Iterator[tuple]:
    """Generate all combinations of target genes and features."""
    for target in target_list:
        for feature in feature_list:
            yield (
                target_masks[target],
                target_masks[reference],
                var_indices[feature],
                target,
                reference,
                feature,
            )


def _batch_generator(
    combinations: Iterator[tuple],
    batch_size: int,
    num_combinations: int,
) -> Iterator[list[tuple]]:
    """Generate batches of combinations."""
    for _i in range(0, num_combinations, batch_size):
        subset = []
        for _ in range(batch_size):
            try:
                subset.append(next(combinations))
            except StopIteration:
                break
        yield subset


def _process_target_batch_shm(
    batch_tasks: list[tuple],
    shm_name: str,
    shape: tuple[int, int],
    dtype: np.dtype,
) -> list[dict[str, float]]:
    """Process a batch of target gene and feature combinations.

    This is the function that is parallelized across multiple workers.
    """
    # Open shared memory once for the batch
    existing_shm = SharedMemory(name=shm_name)
    matrix = np.ndarray(shape=shape, dtype=dtype, buffer=existing_shm.buf)

    results = []
    for (
        target_mask,
        reference_mask,
        var_index,
        target_name,
        reference_name,
        var_name,
    ) in batch_tasks:
        if target_name == reference_name:
            continue

        x_tgt = matrix[target_mask, var_index]
        x_ref = matrix[reference_mask, var_index]

        μ_tgt = np.mean(x_tgt)
        μ_ref = np.mean(x_ref)

        fc = _fold_change(μ_tgt, μ_ref)
        pcc = _percent_change(μ_tgt, μ_ref)
        rs_result = ranksums(x_tgt, x_ref)

        results.append(
            {
                "target": target_name,
                "reference": reference_name,
                "feature": var_name,
                "target_mean": μ_tgt,
                "reference_mean": μ_ref,
                "percent_change": pcc,
                "fold_change": fc,
                "p_value": rs_result.pvalue,
                "statistic": rs_result.statistic,
            }
        )

    existing_shm.close()
    return results


def parallel_differential_expression(
    adata: ad.AnnData,
    groups: list[str] | None = None,
    reference: str = "non-targeting",
    groupby_key: str = "target_gene",
    num_workers: int = 1,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Calculate differential expression between groups of cells.

    Parameters
    ----------
    adata: ad.AnnData
        Annotated data matrix containing gene expression data
    groups: list[str], optional
        List of groups to compare, defaults to None which compares all groups
    reference: str, optional
        Reference group to compare against, defaults to "non-targeting"
    groupby_key: str, optional
        Key in `adata.obs` to group by, defaults to "target_gene"
    num_workers: int
        Number of workers to use for parallel processing, defaults to 1
    batch_size: int
        Number of combinations to process in each batch, defaults to 100

    Returns
    -------
    pd.DataFrame containing differential expression results for each group and feature
    """
    unique_targets = adata.obs[groupby_key].unique()
    if groups is not None:
        unique_targets = [
            target
            for target in unique_targets
            if target in groups or target == reference
        ]
    unique_features = adata.var.index

    # Precompute the number of combinations and batches
    n_combinations = len(unique_targets) * len(unique_features)
    n_batches = n_combinations // batch_size + 1

    # Precompute masks for each target gene
    logger.info("Precomputing masks for each target gene")
    target_masks = {
        target: _get_obs_mask(
            adata=adata, target_name=target, variable_name=groupby_key
        )
        for target in tqdm(unique_targets, desc="Identifying target masks")
    }

    # Precompute variable index for each feature
    logger.info("Precomputing variable indices for each feature")
    var_indices = {
        feature: idx
        for idx, feature in enumerate(
            tqdm(unique_features, desc="Identifying variable indices")
        )
    }

    # Isolate the data matrix from the AnnData object
    logger.info("Creating shared memory memory matrix for parallel computing")
    (shm_name, shape, dtype) = _build_shared_matrix(data=adata.X.toarray())

    logger.info(f"Creating generator of all combinations: N={n_combinations}")
    combinations = _combinations_generator(
        target_masks=target_masks,
        var_indices=var_indices,
        reference=reference,
        target_list=unique_targets,
        feature_list=unique_features,
    )
    logger.info(f"Creating generator of all batches: N={n_batches}")
    batches = _batch_generator(
        combinations=combinations,
        batch_size=batch_size,
        num_combinations=n_combinations,
    )

    # Partial function for parallel processing
    task_fn = partial(
        _process_target_batch_shm,
        shm_name=shm_name,
        shape=shape,
        dtype=dtype,
    )

    logger.info("Initializing parallel processing pool")
    with mp.Pool(num_workers) as pool:
        logger.info("Processing batches")
        batch_results = list(
            tqdm(
                pool.imap(task_fn, batches),
                total=n_batches,
                desc="Processing batches",
            )
        )

    # Flatten results
    logger.info("Flattening results")
    results = [result for batch in batch_results for result in batch]

    # Close shared memory
    logger.info("Closing shared memory pool")
    _conclude_shared_memory(shm_name)

    dataframe = pd.DataFrame(results)
    dataframe["fdr"] = adjust(dataframe["p_value"].values, method="bh")

    return dataframe


def _get_obs_mask(
    adata: ad.AnnData,
    target_name: str,
    variable_name: str = "target_gene",
) -> np.ndarray:
    """Return a boolean mask for a specific target name in the obs variable."""
    return adata.obs[variable_name] == target_name


def _get_var_index(
    adata: ad.AnnData,
    target_gene: str,
) -> int:
    """Return the index of a specific gene in the var variable.

    Raises
    ------
    ValueError
        If the gene is not found in the dataset.
    """
    var_index = np.flatnonzero(adata.var.index == target_gene)
    if len(var_index) == 0:
        raise ValueError(f"Target gene {target_gene} not found in dataset")
    return var_index[0]


def _fold_change(
    μ_tgt: float,
    μ_ref: float,
) -> float:
    """Calculate the fold change between two means."""
    try:
        return μ_tgt / μ_ref
    except ZeroDivisionError:
        return np.nan


def _percent_change(
    μ_tgt: float,
    μ_ref: float,
) -> float:
    """Calculate the percent change between two means."""
    return (μ_tgt - μ_ref) / μ_ref


def vectorized_de(de_results, control_pert, sort_by="abs_fold_change"):
    """
    Create a DataFrame with top k DE genes for each perturbation sorted by the specified metric.

    Parameters
    ----------
    de_results : pd.DataFrame
        DataFrame with differential expression results
    control_pert : str
        Name of the control perturbation
    k : int
        Number of top genes to return for each perturbation
    sort_by : str
        Metric to sort by ('abs_log_fold_change' or 'p_value')

    Returns
    -------
    pd.DataFrame
        DataFrame with rows as perturbations and columns as top K genes
    """
    # Filter out the control perturbation rows
    df = de_results[de_results["target"] != control_pert]

    # Compute absolute fold change (if not already computed)
    df["abs_fold_change"] = df["fold_change"].abs()

    if df[sort_by].dtype == "float16":
        df[sort_by] = df[sort_by].astype("float32")

    # Sort direction depends on metric (descending for fold change, ascending for p-value)
    ascending = True if sort_by == "p_value" else False

    # Sort the DataFrame by target and the chosen metric
    df_sorted = df.sort_values(["target", sort_by], ascending=[True, ascending])

    # For each target, pick the top k rows
    df_sorted["rank"] = df_sorted.groupby("target").cumcount()

    # Pivot the DataFrame so that rows are targets and columns are the ranked top genes
    de_genes = df_sorted.pivot(index="target", columns="rank", values="feature")

    # Optionally, sort the columns so that they are in order from 0 to k-1
    de_genes = de_genes.sort_index(axis=1)

    return de_genes


def vectorized_sig_genes_fc_sort(
    de_results: pd.DataFrame, control_pert: str, pvalue_threshold: float = 0.05
) -> pd.DataFrame:
    df = de_results[de_results["target"] != control_pert].copy()
    df["abs_fold_change"] = df["fold_change"].abs()
    df["abs_fold_change"] = df["abs_fold_change"].fillna(1)

    df["p_value"] = df["p_value"].astype("float32")
    df["abs_fold_change"] = df["abs_fold_change"].astype("float32")

    df = df[df["p_value"] < pvalue_threshold].sort_values(
        ["target", "abs_fold_change"], ascending=[True, False]
    )
    df["rank"] = df.groupby("target").cumcount()
    return df.pivot(index="target", columns="rank", values="feature")
