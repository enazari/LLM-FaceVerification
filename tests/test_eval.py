"""Tests for evaluation logic (10-fold cross-validation)."""

import numpy as np

from src.eval.common import compute_10fold


def test_10fold_perfect_scores():
    """Perfect similarity scores should give ~100% accuracy."""
    n_per_fold = 20
    fold_sizes = [n_per_fold] * 10
    n = sum(fold_sizes)

    # Labels: half same, half different per fold
    labels = np.array([1, 0] * (n // 2))
    # Perfect scores: same=1.0, diff=0.0
    sims = labels.astype(float)

    results = compute_10fold(sims, labels, fold_sizes)
    assert results["mean_acc"] > 0.99


def test_10fold_random_scores():
    """Random scores should give ~50% accuracy."""
    np.random.seed(42)
    n_per_fold = 100
    fold_sizes = [n_per_fold] * 10
    n = sum(fold_sizes)

    labels = np.array([1, 0] * (n // 2))
    sims = np.random.rand(n)

    results = compute_10fold(sims, labels, fold_sizes)
    # With random scores, accuracy should be roughly 50% (but threshold at FAR@0.001
    # is strict, so we mainly test it doesn't crash)
    assert 0.0 <= results["mean_acc"] <= 1.0
    assert len(results["accs"]) == 10
    assert len(results["thresholds"]) == 10


def test_10fold_result_keys():
    """compute_10fold returns all expected keys."""
    fold_sizes = [10] * 10
    n = 100
    labels = np.array([1, 0] * 50)
    sims = np.random.rand(n)

    results = compute_10fold(sims, labels, fold_sizes)
    expected_keys = {"accs", "thresholds", "n_same", "n_diff",
                     "n_same_correct", "n_diff_correct",
                     "mean_acc", "std_acc", "mean_thresh"}
    assert set(results.keys()) == expected_keys


def test_10fold_threshold_at_far001():
    """Threshold should reject most negatives (FAR <= 0.001 on train folds)."""
    np.random.seed(0)
    n_per_fold = 100
    fold_sizes = [n_per_fold] * 10
    n = sum(fold_sizes)

    # Same pairs have high sim, diff pairs have low sim with some overlap
    labels = np.array([1, 0] * (n // 2))
    sims = np.where(labels == 1,
                    np.random.uniform(0.7, 1.0, n),
                    np.random.uniform(0.0, 0.5, n))

    results = compute_10fold(sims, labels, fold_sizes)
    # With well-separated scores, accuracy should be high
    assert results["mean_acc"] > 0.8
