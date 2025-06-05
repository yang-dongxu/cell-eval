"""Array metrics module."""

from typing import Callable, Literal, Sequence

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


def pearson_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute Pearson correlation between mean differences from control."""
    return _generic_evaluation(data, pearsonr, use_delta=True)


def mse(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    return _generic_evaluation(data, skm.mean_squared_error, use_delta=False)


def mae(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    return _generic_evaluation(data, skm.mean_absolute_error, use_delta=False)


def mse_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    return _generic_evaluation(data, skm.mean_squared_error, use_delta=True)


def mae_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    return _generic_evaluation(data, skm.mean_absolute_error, use_delta=True)


def discrimination_score(data: PerturbationAnndataPair) -> dict[str, float]:
    """Compute perturbation discrimination score."""
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=True)
            for d in data.iter_delta_arrays()
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=True)
            for d in data.iter_delta_arrays()
        ]
    )
    gene_names = data.real.var_names
    norm_ranks = {}
    for p_idx, p in enumerate(data.perts):
        include_mask = np.flatnonzero(gene_names != p)
        sim = cosine_similarity(
            real_effects[p_idx, include_mask].reshape(
                1, -1
            ),  # select real effect for current perturbation
            pred_effects[
                :, include_mask
            ],  # compare to all predicted effects across perturbations
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
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for delta_array in data.iter_delta_arrays():
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
        adata: sc.AnnData,
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
        adata: sc.AnnData,
        category_key: str,
        embed_key: str | None = None,
    ) -> sc.AnnData:
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
        np.add.at(centroids, inv, feats)
        centroids /= np.bincount(inv)[:, None]
        adc = sc.AnnData(X=centroids)
        adc.obs[category_key] = uniq
        return adc

    def __call__(self, data: PerturbationAnndataPair) -> float:
        # 1. check same perturbation categories
        real_cats = set(data.real.obs[data.pert_col])
        pred_cats = set(data.pred.obs[data.pert_col])
        if real_cats != pred_cats:
            raise ValueError(
                f"Perturbation categories mismatch:\n"
                f"  only-in-real : {sorted(real_cats - pred_cats)}\n"
                f"  only-in-pred : {sorted(pred_cats - real_cats)}"
            )
        cats_sorted = sorted(real_cats)

        # 2. build centroids
        ad_real_cent = self._centroid_ann(data.real, data.pert_col, self.embed_key)
        ad_pred_cent = self._centroid_ann(data.pred, data.pert_col, self.embed_key)

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
