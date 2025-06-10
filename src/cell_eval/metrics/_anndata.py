"""Array metrics module."""

from typing import Callable, Literal, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as skm
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.pairwise import cosine_similarity

from .._types import PerturbationAnndataPair


def pearson_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute Pearson correlation between mean differences from control."""
    return _generic_evaluation(data, pearsonr, use_delta=True, embed_key=embed_key)


def mse(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=False, embed_key=embed_key
    )


def mae(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=False, embed_key=embed_key
    )


def mse_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=True, embed_key=embed_key
    )


def mae_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=True, embed_key=embed_key
    )


def edistance(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    metric: str = "euclidean",
    **kwargs,
) -> float:
    """Compute Euclidean distance of each perturbation-control delta."""

    def _edistance(
        x: np.ndarray,
        y: np.ndarray,
        metric: str = "euclidean",
        **kwargs,
    ) -> float:
        sigma_x = skm.pairwise_distances(x, metric=metric, **kwargs).mean()
        sigma_y = skm.pairwise_distances(y, metric=metric, **kwargs).mean()
        delta = skm.pairwise_distances(x, y, metric=metric, **kwargs).mean()
        return 2 * delta - sigma_x - sigma_y

    d_real = np.zeros(data.perts.size)
    d_pred = np.zeros(data.perts.size)

    for idx, delta in enumerate(data.iter_delta_arrays(embed_key=embed_key)):
        d_real[idx] = _edistance(
            delta.pert_real, delta.ctrl_real, metric=metric, **kwargs
        )
        if delta.ctrl_pred is None:
            d_pred[idx] = _edistance(
                delta.pert_pred, delta.ctrl_real, metric=metric, **kwargs
            )
        else:
            d_pred[idx] = _edistance(
                delta.pert_pred, delta.ctrl_pred, metric=metric, **kwargs
            )

    return pearsonr(d_real, d_pred).correlation


def discrimination_score_expr(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute perturbation discrimination score using expression data (X) with L1 norm."""
    return _discrimination_score_base(
        data=data, embed_key=None, distance_ord=1, exclude_target_gene=True
    )


def discrimination_score_emb(
    data: PerturbationAnndataPair, embed_key: str = "X_pca"
) -> dict[str, float]:
    """Compute perturbation discrimination score using embedding data with L2 norm."""
    return _discrimination_score_base(
        data=data, embed_key=embed_key, distance_ord=2, exclude_target_gene=False
    )


def _discrimination_score_base(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    distance_ord: int = 1,
    exclude_target_gene: bool = True,
) -> dict[str, float]:
    """Base implementation for discrimination score computation.

    Args:
        data: PerturbationAnndataPair containing real and predicted data
        embed_key: Key for embedding data in obsm, None for expression data
        distance_ord: Order of norm for distance calculation (1 for L1, 2 for L2)
        exclude_target_gene: Whether to exclude target gene from calculation

    Returns:
        Dictionary mapping perturbation names to normalized ranks
    """
    # Compute perturbation effects for all perturbations
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=True)
            for d in data.iter_delta_arrays(embed_key=embed_key)
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=True)
            for d in data.iter_delta_arrays(embed_key=embed_key)
        ]
    )

    norm_ranks = {}
    for p_idx, p in enumerate(data.perts):
        # Determine which features to include in the comparison
        if exclude_target_gene and embed_key is None:
            # For expression data, exclude the target gene
            include_mask = np.flatnonzero(data.genes != p)
        else:
            # For embedding data or when not excluding target gene, use all features
            include_mask = np.ones(real_effects.shape[1], dtype=bool)

        # Get the predicted effect for current perturbation
        pred_effect = pred_effects[p_idx, include_mask]

        # Compute distances to all real effects
        distances = np.array(
            [
                np.linalg.norm(
                    real_effects[i, include_mask] - pred_effect, ord=distance_ord
                )
                for i in range(real_effects.shape[0])
            ]
        )

        # Sort by distance (ascending - lower distance = better match)
        sorted_indices = np.argsort(distances)

        # Find rank of the correct perturbation
        p_index = np.flatnonzero(data.perts == p)[0]
        rank = np.flatnonzero(sorted_indices == p_index)[0]

        # Normalize rank by total number of perturbations
        norm_rank = rank / data.perts.size
        norm_ranks[str(p)] = norm_rank

    return norm_ranks


def discrimination_score(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute perturbation discrimination score."""
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=True)
            for d in data.iter_delta_arrays(embed_key=embed_key)
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=True)
            for d in data.iter_delta_arrays(embed_key=embed_key)
        ]
    )

    norm_ranks = {}
    for p_idx, p in enumerate(data.perts):
        # If no embed key, use gene names to exclude target gene
        if not embed_key:
            include_mask = np.flatnonzero(data.genes != p)
        else:
            include_mask = np.ones(real_effects.shape[1], dtype=bool)

        sim = cosine_similarity(
            real_effects[
                :, include_mask
            ],  # compare to all real effects across perturbations
            pred_effects[p_idx, include_mask].reshape(
                1, -1
            ),  # select pred effect for current perturbation
        ).flatten()
        sorted_rev = np.argsort(sim)[::-1]
        p_index = np.flatnonzero(data.perts == p)[0]
        rank = np.flatnonzero(sorted_rev == p_index)[0]
        norm_rank = rank / data.perts.size
        norm_ranks[str(p)] = norm_rank
    return norm_ranks


def _generic_evaluation(
    data: PerturbationAnndataPair,
    func: Callable[[np.ndarray, np.ndarray], float],
    use_delta: bool = False,
    embed_key: str | None = None,
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for delta_array in data.iter_delta_arrays(embed_key=embed_key):
        if use_delta:
            x = delta_array.perturbation_effect(which="pred", abs=False)
            y = delta_array.perturbation_effect(which="real", abs=False)
        else:
            x = delta_array.pert_pred.mean(axis=0)
            y = delta_array.pert_real.mean(axis=0)

        result = func(x, y)
        if isinstance(result, tuple):
            result = result[0]

        res[delta_array.pert] = float(result)

    return res


class ClusteringAgreement:
    """Compute clustering agreement between real and predicted perturbation centroids."""

    def __init__(
        self,
        embed_key: str | None = None,
        real_resolution: float = 1.0,
        pred_resolutions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: Literal["ami", "nmi", "ari"] = "ami",
        n_neighbors: int = 15,
    ) -> None:
        self.embed_key = embed_key
        self.real_resolution = real_resolution
        self.pred_resolutions = pred_resolutions
        self.metric = metric
        self.n_neighbors = n_neighbors

    @staticmethod
    def _score(
        labels_real: Sequence[int],
        labels_pred: Sequence[int],
        metric: Literal["ami", "nmi", "ari"],
    ) -> float:
        if metric == "ami":
            return adjusted_mutual_info_score(labels_real, labels_pred)
        if metric == "nmi":
            return normalized_mutual_info_score(labels_real, labels_pred)
        if metric == "ari":
            return (adjusted_rand_score(labels_real, labels_pred) + 1) / 2
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _cluster_leiden(
        adata: ad.AnnData,
        resolution: float,
        key_added: str,
        n_neighbors: int = 15,
    ) -> None:
        if key_added in adata.obs:
            return
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(
                adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep="X"
            )
        sc.tl.leiden(adata, resolution=resolution, key_added=key_added)

    @staticmethod
    def _centroid_ann(
        adata: ad.AnnData,
        category_key: str,
        control_pert: str,
        embed_key: str | None = None,
    ) -> ad.AnnData:
        # Isolate the features
        feats = adata.obsm.get(embed_key, adata.X)

        # Convert to float if not already
        if feats.dtype != np.dtype("float64"):
            feats = feats.astype(np.float64)

        # Densify if required
        if issparse(feats):
            feats = feats.toarray()

        cats = adata.obs[category_key].values
        uniq, inv = np.unique(cats, return_inverse=True)
        centroids = np.zeros((uniq.size, feats.shape[1]), dtype=feats.dtype)

        for i, cat in enumerate(uniq):
            mask = cats == cat
            if np.any(mask):
                centroids[i] = feats[mask].mean(axis=0)

        adc = ad.AnnData(X=centroids)
        adc.obs[category_key] = uniq
        return adc[adc.obs[category_key] != control_pert]

    def __call__(self, data: PerturbationAnndataPair) -> float:
        cats_sorted = sorted([c for c in data.perts if c != data.control_pert])

        # 2. build centroids
        ad_real_cent = self._centroid_ann(
            adata=data.real,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )
        ad_pred_cent = self._centroid_ann(
            adata=data.pred,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )

        # 3. cluster real once
        real_key = "real_clusters"
        self._cluster_leiden(
            ad_real_cent, self.real_resolution, real_key, self.n_neighbors
        )
        ad_real_cent.obs = ad_real_cent.obs.set_index(data.pert_col).loc[cats_sorted]
        real_labels = pd.Categorical(ad_real_cent.obs[real_key])

        # 4. sweep predicted resolutions
        best_score = 0.0
        ad_pred_cent.obs = ad_pred_cent.obs.set_index(data.pert_col).loc[cats_sorted]
        for r in self.pred_resolutions:
            pred_key = f"pred_clusters_{r}"
            self._cluster_leiden(ad_pred_cent, r, pred_key, self.n_neighbors)
            pred_labels = pd.Categorical(ad_pred_cent.obs[pred_key])
            score = self._score(real_labels, pred_labels, self.metric)
            best_score = max(best_score, score)

        return float(best_score)
