"""Statistical utilities for ablation comparison.

Pure numpy implementations of:
  - bootstrap_ci(scores, ci=0.95, n_boot=1000): bootstrap confidence interval
    of the sample mean.
  - paired_permutation_test(scores_a, scores_b, n_perm=10000): two-sided
    paired permutation test on the difference of means.

Both functions support an optional ``rng_seed`` for reproducibility.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def _as_array(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D iterable, got shape {arr.shape}")
    return arr


def bootstrap_ci(
    scores: Iterable[float],
    ci: float = 0.95,
    n_boot: int = 1000,
    rng_seed: Optional[int] = None,
    statistic: str = "mean",
) -> dict:
    """Bootstrap confidence interval of a sample statistic.

    Args:
        scores:    1-D iterable of float scores (e.g., per-seed accuracies).
        ci:        Confidence level in (0, 1), e.g. 0.95 for a 95% CI.
        n_boot:    Number of bootstrap resamples.
        rng_seed:  Optional seed for the numpy RNG used during resampling.
        statistic: 'mean' or 'median'.

    Returns a dict with keys:
        - point:   the statistic on the original sample
        - lower:   lower CI bound
        - upper:   upper CI bound
        - n:       sample size
        - n_boot:  number of bootstrap iterations actually used
        - ci:      requested confidence level
    """
    arr = _as_array(scores)
    if arr.size == 0:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "n": 0, "n_boot": int(n_boot), "ci": float(ci)}
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if int(n_boot) < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if statistic not in {"mean", "median"}:
        raise ValueError(f"Unknown statistic: {statistic}")

    rng = np.random.default_rng(rng_seed)
    n = arr.size
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        sample_idx = rng.integers(0, n, size=n)
        sample = arr[sample_idx]
        boots[i] = sample.mean() if statistic == "mean" else np.median(sample)

    alpha = 1.0 - float(ci)
    low_q = alpha / 2.0
    high_q = 1.0 - alpha / 2.0
    lower = float(np.quantile(boots, low_q))
    upper = float(np.quantile(boots, high_q))
    point = float(arr.mean() if statistic == "mean" else np.median(arr))
    return {"point": point, "lower": lower, "upper": upper,
            "n": int(n), "n_boot": int(n_boot), "ci": float(ci)}


def paired_permutation_test(
    scores_a: Iterable[float],
    scores_b: Iterable[float],
    n_perm: int = 10000,
    rng_seed: Optional[int] = None,
    two_sided: bool = True,
) -> dict:
    """Paired permutation test on the difference of means (scores_a -
    scores_b). At each of ``n_perm`` iterations we independently flip the
    sign of each paired difference with probability 0.5 and measure the
    permuted statistic. The p-value is the fraction of permuted statistics
    whose magnitude is at least as extreme as the observed one.

    Args:
        scores_a / scores_b: equal-length 1-D iterables of paired scores.
        n_perm:    Number of random sign flips.
        rng_seed:  Optional seed.
        two_sided: If True (default), returns a two-sided p-value; else a
                   one-sided test for the alternative ``mean(a) > mean(b)``.

    Returns a dict with keys:
        - observed_diff:  mean(a) - mean(b) on the original sample
        - p_value:        permutation p-value
        - n:              number of pairs
        - n_perm:         number of permutations actually used
        - two_sided:      bool flag for clarity
    """
    a = _as_array(scores_a)
    b = _as_array(scores_b)
    if a.size != b.size:
        raise ValueError(f"Paired test requires equal sizes: got {a.size} vs {b.size}")
    if a.size == 0:
        return {"observed_diff": float("nan"), "p_value": float("nan"),
                "n": 0, "n_perm": int(n_perm), "two_sided": bool(two_sided)}
    if int(n_perm) < 1:
        raise ValueError(f"n_perm must be >= 1, got {n_perm}")

    diff = a - b
    observed = float(diff.mean())
    rng = np.random.default_rng(rng_seed)

    # Vectorize sign flipping in chunks to keep memory bounded for large n_perm.
    n = diff.size
    n_perm = int(n_perm)
    chunk = max(1, min(n_perm, 2048))
    extreme = 0
    done = 0
    while done < n_perm:
        cur = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(cur, n))
        perm_means = (signs * diff[np.newaxis, :]).mean(axis=1)
        if two_sided:
            extreme += int(np.sum(np.abs(perm_means) >= abs(observed) - 1e-12))
        else:
            extreme += int(np.sum(perm_means >= observed - 1e-12))
        done += cur

    # +1 / +1 smoothing keeps p-value strictly in (0, 1] and matches the
    # exchangeability argument under the null.
    p_value = (extreme + 1) / (n_perm + 1)
    return {"observed_diff": observed, "p_value": float(p_value),
            "n": int(n), "n_perm": int(n_perm), "two_sided": bool(two_sided)}


def diff_ci(
    scores_a: Iterable[float],
    scores_b: Iterable[float],
    ci: float = 0.95,
    n_boot: int = 1000,
    rng_seed: Optional[int] = None,
) -> dict:
    """Bootstrap CI for the paired difference mean(scores_a) - mean(scores_b).
    Convenient wrapper used by the aggregator."""
    a = _as_array(scores_a)
    b = _as_array(scores_b)
    if a.size != b.size:
        raise ValueError(f"diff_ci requires paired samples of equal size: got {a.size} vs {b.size}")
    diff = a - b
    result = bootstrap_ci(diff, ci=ci, n_boot=n_boot, rng_seed=rng_seed, statistic="mean")
    result["mean_a"] = float(a.mean()) if a.size else float("nan")
    result["mean_b"] = float(b.mean()) if b.size else float("nan")
    return result


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    truth_mean = 0.7
    samples = truth_mean + 0.05 * rng.standard_normal(10)
    ci = bootstrap_ci(samples, ci=0.95, n_boot=2000, rng_seed=0)
    print("bootstrap_ci:", ci)

    a = truth_mean + 0.05 * rng.standard_normal(10)
    b = truth_mean + 0.05 * rng.standard_normal(10)  # same distribution
    perm = paired_permutation_test(a, b, n_perm=5000, rng_seed=0)
    print("paired_permutation_test (null):", perm)

    a2 = a.copy()
    b2 = a2 - 0.10  # systematic shift
    perm2 = paired_permutation_test(a2, b2, n_perm=5000, rng_seed=0)
    print("paired_permutation_test (shifted):", perm2)
    diff = diff_ci(a2, b2, ci=0.95, n_boot=2000, rng_seed=0)
    print("diff_ci (shifted):", diff)
