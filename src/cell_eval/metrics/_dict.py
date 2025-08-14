"""Utilities for working with dictionary-based perturbation representations."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import sklearn.metrics as skm


def discrimination_score_dict(
    pred: Mapping[str, np.ndarray],
    real: Mapping[str, np.ndarray],
    metric: str = "l1",
) -> dict[str, float]:
    """Compute discrimination score from dictionaries of expression arrays.

    Args:
        pred: Mapping of perturbation names to predicted expression vectors.
        real: Mapping of perturbation names to ground truth expression vectors.
        metric: Metric for distance calculation (e.g., "l1", "l2", "cosine").

    Returns:
        Dictionary mapping perturbation names to normalized ranks where 1 is best
        and values near 0 indicate poor discrimination.

    Raises:
        ValueError: If the keys of ``pred`` and ``real`` do not match.
    """
    if set(pred.keys()) != set(real.keys()):
        raise ValueError("pred and real must contain the same perturbation keys")

    keys = list(real.keys())
    real_matrix = np.vstack([real[k] for k in keys])
    pred_matrix = np.vstack([pred[k] for k in keys])

    scores: dict[str, float] = {}
    for idx, key in enumerate(keys):
        distances = skm.pairwise_distances(
            real_matrix,
            pred_matrix[idx].reshape(1, -1),
            metric=metric,
        ).flatten()
        sorted_indices = np.argsort(distances)
        rank = np.flatnonzero(sorted_indices == idx)[0]
        norm_rank = rank / len(keys)
        scores[str(key)] = 1 - norm_rank

    return scores
