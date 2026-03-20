"""Shared 10-fold cross-validation logic for face verification benchmarks."""

import csv
import os

import numpy as np


def compute_10fold(sims, labels, fold_sizes):
    """Run 10-fold cross-validation at FAR <= 0.001.

    Args:
        sims: np.ndarray of similarity scores (cosine sim or P(Yes)).
        labels: np.ndarray of ground-truth labels (1=same, 0=diff).
        fold_sizes: list of int, number of pairs per fold.

    Returns dict with keys: accs, thresholds, n_same, n_diff,
        n_same_correct, n_diff_correct, mean_acc, std_acc, mean_thresh.
    """
    n = len(labels)
    fold_starts = np.cumsum([0] + fold_sizes)
    accs, thresholds = [], []
    n_same_correct_list, n_diff_correct_list = [], []
    n_same_list, n_diff_list = [], []

    for i in range(10):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[fold_starts[i]:fold_starts[i + 1]] = True
        train_mask = ~test_mask

        # Threshold at FAR <= 0.001 from train folds
        train_diff_sims = sims[train_mask & (labels == 0)]
        sorted_diffs = np.sort(train_diff_sims)[::-1]
        max_fa = int(np.floor(len(sorted_diffs) * 0.001))
        if max_fa == 0:
            thresh = float(sorted_diffs[0]) + 1e-6
        else:
            thresh = float(sorted_diffs[max_fa - 1])

        test_sims = sims[test_mask]
        test_labels = labels[test_mask]
        preds = (test_sims >= thresh).astype(int)
        acc = (preds == test_labels).mean()

        same_mask = test_labels == 1
        diff_mask = test_labels == 0
        n_same_list.append(int(same_mask.sum()))
        n_diff_list.append(int(diff_mask.sum()))
        n_same_correct_list.append(int((preds[same_mask] == 1).sum()))
        n_diff_correct_list.append(int((preds[diff_mask] == 0).sum()))

        accs.append(acc)
        thresholds.append(thresh)
        print(f"  fold {i+1}: acc={acc:.4f}  thresh={thresh:.4f}")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_thresh = np.mean(thresholds)
    print(f"  mean: acc={mean_acc:.4f} ± {std_acc:.4f}  thresh={mean_thresh:.4f}")

    return {
        "accs": accs,
        "thresholds": thresholds,
        "n_same": n_same_list,
        "n_diff": n_diff_list,
        "n_same_correct": n_same_correct_list,
        "n_diff_correct": n_diff_correct_list,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_thresh": mean_thresh,
    }


def write_results_csv(output_dir, filename, results):
    """Write per-fold results CSV."""
    csv_path = os.path.join(output_dir, filename)
    accs = results["accs"]
    thresholds = results["thresholds"]
    n_same = results["n_same"]
    n_diff = results["n_diff"]
    n_same_correct = results["n_same_correct"]
    n_diff_correct = results["n_diff_correct"]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "accuracy", "threshold",
                     "n_same", "same_correct", "TAR",
                     "n_diff", "diff_correct", "1-FAR"])
        for i in range(10):
            tar = n_same_correct[i] / n_same[i] if n_same[i] else 0
            spec = n_diff_correct[i] / n_diff[i] if n_diff[i] else 0
            w.writerow([
                i + 1,
                f"{accs[i]:.6f}", f"{thresholds[i]:.6f}",
                n_same[i], n_same_correct[i], f"{tar:.6f}",
                n_diff[i], n_diff_correct[i], f"{spec:.6f}",
            ])
        w.writerow([
            "mean",
            f"{results['mean_acc']:.6f}", f"{results['mean_thresh']:.6f}",
            "", "", f"{np.mean([sc/ns for sc, ns in zip(n_same_correct, n_same)]):.6f}",
            "", "", f"{np.mean([dc/nd for dc, nd in zip(n_diff_correct, n_diff)]):.6f}",
        ])
        w.writerow(["std", f"{results['std_acc']:.6f}", "", "", "", "", "", "", ""])
    print(f"  saved → {csv_path}")
