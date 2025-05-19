import logging
import os
import time
from typing import Optional

import anndata as ad
import pandas as pd
from pdex import parallel_differential_expression

# Configure logger
tools_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parallel_compute_de(
    adata_gene: ad.AnnData,
    control_pert: str,
    pert_col: str,
    outdir: Optional[str] = None,
    split: str = "real",
    prefix: str = "",
    n_threads: int = 1,
    batch_size: int = 1000,
    metric: str = "wilcoxon",
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
    outdir : str, optional
        Directory to save output files, by default None
    split : str, optional
        Split type for differential expression analysis, by default "real"
    prefix : str, optional
        Prefix for output file names, by default ""
    n_threads : int, optional
        Number of threads to use for parallel computation, by default 1
    batch_size : int, optional
        Batch size for parallel computation, by default 1000
    metric: str
        Metric to use when computing differential expression [wilcoxon, anderson, t-test]

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
        metric=metric,
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

    df = df[df["fdr"] < pvalue_threshold].sort_values(
        ["target", "abs_fold_change"], ascending=[True, False]
    )
    df["rank"] = df.groupby("target").cumcount()
    return df.pivot(index="target", columns="rank", values="feature")
