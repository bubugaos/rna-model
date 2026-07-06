"""Aggregate evaluation results across multi-seed multi-mode runs.

Reads the ``manifest.json`` produced by :mod:`scripts.run_multiseed`, locates
the ``evaluate_tasks_result*_summary.json`` written by
:mod:`tasks.evaluate_tasks` inside each run's ``exp_dir``, aggregates the
scalar metrics (currently classification accuracy and macro_f1) per chunking
mode, and applies :mod:`utils.stats_utils` to produce:

  - per-mode mean ± std and bootstrap confidence intervals
  - pairwise paired permutation tests across modes
  - a markdown table summary

Outputs:
  - ``<output_dir>/summary.json``: machine-readable
  - ``<output_dir>/summary.md``:   human-readable

Usage:
    python -m utils.aggregate_results --manifest path/to/manifest.json \
        --metrics accuracy,macro_f1

For smoke verification it also supports a ``--smoke`` flag that synthesizes a
small dummy manifest in a temp dir and runs the full aggregation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.stats_utils import bootstrap_ci, paired_permutation_test, diff_ci  # noqa: E402


_EVAL_SUMMARY_CANDIDATES = (
    "evaluate_tasks_result_summary.json",
    "evaluate_tasks_result_fixed_summary.json",
    "evaluate_tasks_result_noChunk_summary.json",
)


def _find_summary_for_run(run: Dict[str, Any]) -> Optional[str]:
    exp_dir = run.get("exp_dir")
    if not exp_dir or not os.path.isdir(exp_dir):
        return None
    for name in _EVAL_SUMMARY_CANDIDATES:
        path = os.path.join(exp_dir, name)
        if os.path.exists(path):
            return path
    return None


def _extract_metric(summary: Dict[str, Any], metric: str) -> Optional[float]:
    """Extract a scalar metric from a summary dict, e.g. 'accuracy' or
    'classification.accuracy' (defaulting to the classification subtree)."""
    if "." in metric:
        keys = metric.split(".")
    else:
        keys = ["classification", metric]
    node: Any = summary
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def collect_metric_scores(
    manifest: Dict[str, Any],
    metric: str,
) -> Dict[str, Dict[int, float]]:
    """Return ``{mode: {seed: score}}`` for the given metric, skipping runs
    whose evaluate summary is missing or the metric is absent."""
    out: Dict[str, Dict[int, float]] = {}
    for run in manifest.get("runs", []):
        mode = run.get("mode")
        seed = run.get("seed")
        if mode is None or seed is None:
            continue
        summary_path = _find_summary_for_run(run)
        if summary_path is None:
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue
        val = _extract_metric(summary, metric)
        if val is None:
            continue
        out.setdefault(mode, {})[int(seed)] = float(val)
    return out


def aggregate(
    manifest: Dict[str, Any],
    metrics: Sequence[str],
    ci: float = 0.95,
    n_boot: int = 1000,
    n_perm: int = 10000,
    rng_seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Compute per-mode mean/std/CI and pairwise permutation tests for every
    metric in ``metrics``."""
    result: Dict[str, Any] = {
        "metrics": list(metrics),
        "ci": float(ci),
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "per_metric": {},
    }

    for metric in metrics:
        per_mode = collect_metric_scores(manifest, metric)
        per_metric: Dict[str, Any] = {"per_mode": {}, "pairwise": {}}

        seed_sets = [set(v.keys()) for v in per_mode.values()]
        shared_seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []

        for mode, seed_to_score in per_mode.items():
            arr = np.array([seed_to_score[s] for s in sorted(seed_to_score.keys())], dtype=np.float64)
            if arr.size:
                ci_dict = bootstrap_ci(arr, ci=ci, n_boot=n_boot, rng_seed=rng_seed)
                per_metric["per_mode"][mode] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "n": int(arr.size),
                    "seeds": sorted(seed_to_score.keys()),
                    "ci_lower": ci_dict["lower"],
                    "ci_upper": ci_dict["upper"],
                }
            else:
                per_metric["per_mode"][mode] = {
                    "mean": float("nan"), "std": 0.0, "n": 0, "seeds": [],
                    "ci_lower": float("nan"), "ci_upper": float("nan"),
                }

        modes = sorted(per_mode.keys())
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                a_mode, b_mode = modes[i], modes[j]
                if not shared_seeds:
                    continue
                a_scores = [per_mode[a_mode][s] for s in shared_seeds]
                b_scores = [per_mode[b_mode][s] for s in shared_seeds]
                perm = paired_permutation_test(a_scores, b_scores, n_perm=n_perm, rng_seed=rng_seed)
                dci = diff_ci(a_scores, b_scores, ci=ci, n_boot=n_boot, rng_seed=rng_seed)
                pair_key = f"{a_mode}_vs_{b_mode}"
                per_metric["pairwise"][pair_key] = {
                    "shared_seeds": shared_seeds,
                    "observed_diff": perm["observed_diff"],
                    "p_value": perm["p_value"],
                    "diff_ci_lower": dci["lower"],
                    "diff_ci_upper": dci["upper"],
                }

        result["per_metric"][metric] = per_metric

    return result


def format_markdown(agg: Dict[str, Any]) -> str:
    """Render aggregation result as a markdown report."""
    lines: List[str] = []
    lines.append("# Multi-seed Aggregation Summary")
    lines.append("")
    for metric, body in agg.get("per_metric", {}).items():
        lines.append(f"## metric: `{metric}`")
        lines.append("")
        lines.append("### Per-mode (mean +/- std, bootstrap CI)")
        lines.append("")
        lines.append("| mode | n | seeds | mean | std | CI lower | CI upper |")
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for mode in sorted(body.get("per_mode", {}).keys()):
            r = body["per_mode"][mode]
            seeds_str = ",".join(str(s) for s in r.get("seeds", []))
            lines.append(
                f"| {mode} | {r['n']} | {seeds_str} | "
                f"{r['mean']:.4f} | {r['std']:.4f} | "
                f"{r['ci_lower']:.4f} | {r['ci_upper']:.4f} |"
            )
        lines.append("")
        pairwise = body.get("pairwise", {})
        if pairwise:
            lines.append("### Pairwise paired permutation test")
            lines.append("")
            lines.append("| comparison | shared seeds | diff (a - b) | diff CI | p-value |")
            lines.append("|---|---|---:|---|---:|")
            for pair_key in sorted(pairwise.keys()):
                pr = pairwise[pair_key]
                seeds_str = ",".join(str(s) for s in pr.get("shared_seeds", []))
                ci_str = f"[{pr['diff_ci_lower']:.4f}, {pr['diff_ci_upper']:.4f}]"
                lines.append(
                    f"| {pair_key} | {seeds_str} | {pr['observed_diff']:.4f} | "
                    f"{ci_str} | {pr['p_value']:.4f} |"
                )
            lines.append("")
    return "\n".join(lines)


def _build_smoke_manifest(work_dir: str) -> str:
    """Synthesize a small but realistic manifest + per-run evaluate summaries
    so the aggregator can be exercised without a real training run."""
    runs = []
    for mode_idx, (mode, base_acc) in enumerate([
        ("no_chunk", 0.50),
        ("fixed",    0.62),
        ("dynamic",  0.65),
    ]):
        for seed in (42, 43, 44):
            exp_name = f"smoke_{mode}_seed{seed}"
            exp_dir = os.path.join(work_dir, "experiments", exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            # tiny random jitter per seed/mode so stats are non-degenerate
            jitter = ((seed - 42) * 0.01) + (mode_idx * 0.005)
            acc = base_acc + jitter
            f1 = acc - 0.02
            summary = {
                "exp_name": exp_name,
                "use_no_chunk": (mode == "no_chunk"),
                "use_fixed_router": (mode == "fixed"),
                "classification": {
                    "accuracy": float(acc),
                    "macro_f1": float(f1),
                    "expected_segments": float(1 if mode == "no_chunk" else 16),
                    "boundary_p_mean": float(0.0 if mode == "no_chunk" else 0.3),
                    "per_bucket": {},
                },
                "boundary_ablation": {},
            }
            if mode == "no_chunk":
                fname = "evaluate_tasks_result_noChunk_summary.json"
            elif mode == "fixed":
                fname = "evaluate_tasks_result_fixed_summary.json"
            else:
                fname = "evaluate_tasks_result_summary.json"
            with open(os.path.join(exp_dir, fname), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            runs.append({"mode": mode, "seed": seed, "exp_name": exp_name,
                         "exp_dir": exp_dir, "phases": {
                             "pretrain": {"ok": True}, "finetune": {"ok": True}, "evaluate": {"ok": True},
                         }})
    manifest = {
        "created_at": "smoke",
        "base_dir": work_dir,
        "base_name": "smoke",
        "modes": ["no_chunk", "fixed", "dynamic"],
        "seeds": [42, 43, 44],
        "smoke": True,
        "output_dir": work_dir,
        "runs": runs,
    }
    manifest_path = os.path.join(work_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate multi-seed multi-mode evaluation results")
    p.add_argument("--manifest", type=str, default=None,
                   help="Path to manifest.json produced by scripts/run_multiseed.py")
    p.add_argument("--metrics", type=str, default="accuracy,macro_f1",
                   help="Comma-separated metrics to aggregate. Each may be a "
                        "bare key like 'accuracy' (interpreted as "
                        "'classification.accuracy') or a dotted path.")
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--rng_seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write summary.json and summary.md. "
                        "Default: same dir as manifest.")
    p.add_argument("--smoke", action="store_true",
                   help="Run end-to-end on a synthesized manifest in a temp directory.")
    return p.parse_args()


def main():
    args = parse_args()

    cleanup_dir: Optional[str] = None
    if args.smoke:
        tmp_dir = tempfile.mkdtemp(prefix="rna_smoke_aggregate_")
        manifest_path = _build_smoke_manifest(tmp_dir)
        cleanup_dir = tmp_dir
        if args.output_dir is None:
            args.output_dir = tmp_dir
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required (or pass --smoke).")
        manifest_path = args.manifest

    if not os.path.exists(manifest_path):
        raise SystemExit(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified.")

    agg = aggregate(manifest, metrics=metrics, ci=args.ci, n_boot=args.n_boot,
                    n_perm=args.n_perm, rng_seed=args.rng_seed)

    output_dir = args.output_dir or os.path.dirname(manifest_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    summary_json = os.path.join(output_dir, "summary.json")
    summary_md = os.path.join(output_dir, "summary.md")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    md_text = format_markdown(agg)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(md_text)
    print(f"\nWrote {summary_json} and {summary_md}")

    if cleanup_dir is not None:
        print(f"(smoke artifacts retained at {cleanup_dir} for inspection)")


if __name__ == "__main__":
    main()
