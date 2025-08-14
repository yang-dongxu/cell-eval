import numpy as np
import pytest

from cell_eval.metrics import discrimination_score_dict


def test_discrimination_score_dict_perfect():
    real = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    pred = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    scores = discrimination_score_dict(pred, real, metric="l2")
    assert scores["a"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(1.0)


def test_discrimination_score_dict_swapped():
    real = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    pred = {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 0.0])}
    scores = discrimination_score_dict(pred, real, metric="l2")
    assert scores["a"] == pytest.approx(0.5)
    assert scores["b"] == pytest.approx(0.5)
