import anndata as ad


def split_anndata_on_celltype(
    adata: ad.AnnData,
    celltype_col: str,
) -> dict[str, ad.AnnData]:
    """Split anndata on celltype column.

    Args:
        adata: AnnData object
        celltype_col: Column name in adata.obs that contains the celltype labels

    Returns:
        dict[str, AnnData]: Dictionary of AnnData objects, keyed by celltype
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(
            f"Celltype column {celltype_col} not found in adata.obs: {adata.obs.columns}"
        )

    return {
        ct: adata[adata.obs[celltype_col] == ct]
        for ct in adata.obs[celltype_col].unique()
    }
