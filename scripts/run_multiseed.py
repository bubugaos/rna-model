"""Multi-seed multi-mode experiment driver.

Runs the three-stage RNA pipeline (pretrain -> finetune -> evaluate) for every
combination of ``--modes`` and ``--seeds`` and writes a structured
``manifest.json`` that downstream aggregation (``utils/aggregate_results.py``)
can consume.

Each (mode, seed) gets a unique ``exp_name`` of the form::

    {base_name}_{mode}_seed{seed}

so different ablation modes never share the same exp_dir.

Usage (smoke, fast):
    python scripts/run_multiseed.py --smoke --seeds 42,43

Usage (real run):
    python scripts/run_multiseed.py \
        --base_dir /path/to/RNA-model \
        --base_name rfam_run \
        --modes no_chunk,fixed,dynamic \
        --seeds 42,43,44 \
        --pretrain_epochs 50 \
        --finetune_epochs 30
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _parse_csv(text: str) -> List[str]:
    if text is None:
        return []
    return [p.strip() for p in str(text).split(",") if p.strip() != ""]


def _mode_to_cli_flags(mode: str, chunk_size: int) -> List[str]:
    if mode == "no_chunk":
        return ["--no_chunk"]
    if mode == "fixed":
        return ["--fixed_router", "--chunk_size", str(chunk_size)]
    if mode == "dynamic":
        return []
    raise ValueError(f"Unknown chunking mode: {mode!r}. Allowed: no_chunk, fixed, dynamic.")


def _run(cmd: Sequence[str], log_path: Optional[str] = None) -> Dict:
    """Run a subprocess and capture exit code + stdout/stderr.

    Returns a dict ``{ok, exit_code, log_path?, stdout_tail, stderr_tail}``.
    Output is tee-d to ``log_path`` if provided.
    """
    print(f"[run_multiseed] $ {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as e:
        return {"ok": False, "exit_code": -1, "stdout_tail": "", "stderr_tail": str(e)}

    out = proc.stdout or ""
    err = proc.stderr or ""
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n\n")
            f.write("--- STDOUT ---\n")
            f.write(out)
            f.write("\n--- STDERR ---\n")
            f.write(err)

    return {
        "ok": proc.returncode == 0,
        "exit_code": int(proc.returncode),
        "log_path": log_path,
        "stdout_tail": _tail_lines(out, 30),
        "stderr_tail": _tail_lines(err, 30),
    }


def _tail_lines(text: str, n: int) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _dataloader_flags(args: argparse.Namespace) -> List[str]:
    """Build CLI flags for DataLoader params shared by all stages."""
    flags: List[str] = []
    if getattr(args, "num_workers", None) is not None:
        flags += ["--num_workers", str(args.num_workers)]
    if getattr(args, "pin_memory", False):
        flags += ["--pin_memory"]
    if getattr(args, "persistent_workers", False):
        flags += ["--persistent_workers"]
    return flags


def _do_one_run(
    mode: str,
    seed: int,
    args: argparse.Namespace,
) -> Dict:
    base_name = args.base_name
    exp_name = f"{base_name}_{mode}_seed{seed}"
    exp_dir = os.path.join(args.base_dir, "experiments", base_name, exp_name)
    mode_flags = _mode_to_cli_flags(mode, args.chunk_size)
    dl_flags = _dataloader_flags(args)
    run_record: Dict = {
        "mode": mode,
        "seed": int(seed),
        "exp_name": exp_name,
        "exp_dir": exp_dir,
        "phases": {},
    }

    # ── 1. Pretrain ──
    pretrain_cmd: List[str] = [
        sys.executable, "-m", "src.training.train",
        "--exp_name", exp_name,
        "--exp_dir", exp_dir,
        "--seed", str(seed),
    ]
    pretrain_cmd += mode_flags
    pretrain_cmd += dl_flags
    if args.bppm_mode is not None:
        pretrain_cmd += ["--bppm_mode", str(args.bppm_mode)]
    if args.cache_dir is not None:
        pretrain_cmd += ["--cache_dir", str(args.cache_dir)]
    if getattr(args, "prefetch_workers", None) is not None:
        pretrain_cmd += ["--prefetch_workers", str(args.prefetch_workers)]
    if args.smoke:
        pretrain_cmd += ["--smoke"]
    else:
        if args.pretrain_epochs is not None:
            pretrain_cmd += ["--num_epochs", str(args.pretrain_epochs)]
        if args.pretrain_max_steps is not None:
            pretrain_cmd += ["--max_steps", str(args.pretrain_max_steps)]
    pretrain_log = os.path.join(exp_dir, f"{exp_name}_pretrain.log")
    run_record["phases"]["pretrain"] = _run(pretrain_cmd, log_path=pretrain_log)

    # If pretrain failed, skip finetune/eval to avoid useless work.
    if not run_record["phases"]["pretrain"]["ok"]:
        run_record["phases"]["finetune"] = {"ok": False, "exit_code": None, "skipped": True}
        run_record["phases"]["evaluate"] = {"ok": False, "exit_code": None, "skipped": True}
        return run_record

    # ── 2. Finetune ──
    finetune_cmd: List[str] = [
        sys.executable, "tasks/finetune_classifier.py",
        "--exp_name", exp_name,
        "--exp_dir", exp_dir,
        "--base_dir", args.base_dir,
    ]
    finetune_cmd += mode_flags
    finetune_cmd += dl_flags
    if args.bppm_mode is not None:
        finetune_cmd += ["--bppm_mode", str(args.bppm_mode)]
    if args.cache_dir is not None:
        finetune_cmd += ["--cache_dir", str(args.cache_dir)]
    if args.smoke:
        finetune_cmd += ["--smoke"]
    else:
        if args.finetune_epochs is not None:
            finetune_cmd += ["--epochs", str(args.finetune_epochs)]
        if args.finetune_max_steps is not None:
            finetune_cmd += ["--max_steps", str(args.finetune_max_steps)]
    finetune_log = os.path.join(exp_dir, f"{exp_name}_finetune.log")
    run_record["phases"]["finetune"] = _run(finetune_cmd, log_path=finetune_log)

    if not run_record["phases"]["finetune"]["ok"]:
        run_record["phases"]["evaluate"] = {"ok": False, "exit_code": None, "skipped": True}
        return run_record

    # ── 3. Evaluate ──
    evaluate_cmd: List[str] = [
        sys.executable, "tasks/evaluate_tasks.py",
        "--exp_name", exp_name,
        "--exp_dir", exp_dir,
        "--base_dir", args.base_dir,
    ]
    evaluate_cmd += mode_flags
    evaluate_cmd += dl_flags
    if args.bppm_mode is not None:
        evaluate_cmd += ["--bppm_mode", str(args.bppm_mode)]
    if args.cache_dir is not None:
        evaluate_cmd += ["--cache_dir", str(args.cache_dir)]
    if args.skip_boundary:
        evaluate_cmd += ["--skip_boundary"]
    if args.smoke:
        evaluate_cmd += ["--smoke"]
    else:
        if args.eval_max_steps is not None:
            evaluate_cmd += ["--max_steps", str(args.eval_max_steps)]
    eval_log = os.path.join(exp_dir, f"{exp_name}_evaluate.log")
    run_record["phases"]["evaluate"] = _run(evaluate_cmd, log_path=eval_log)

    return run_record


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed multi-mode RNA pipeline driver")
    p.add_argument("--base_dir", type=str, default=_PROJECT_ROOT,
                   help="Base directory used as data root and experiments root.")
    p.add_argument("--base_name", type=str, default="rna_multiseed",
                   help="Base experiment name; mode + seed get appended.")
    p.add_argument("--modes", type=str, default="dynamic,no_chunk,fixed",
                   help="Comma-separated chunking modes.")
    p.add_argument("--seeds", type=str, default="42,43,44",
                   help="Comma-separated integer seeds.")
    p.add_argument("--chunk_size", type=int, default=8,
                   help="Chunk size for the fixed router.")
    p.add_argument("--pretrain_epochs", type=int, default=None)
    p.add_argument("--pretrain_max_steps", type=int, default=None)
    p.add_argument("--finetune_epochs", type=int, default=None)
    p.add_argument("--finetune_max_steps", type=int, default=None)
    p.add_argument("--eval_max_steps", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Forward --smoke to each child stage; runs in seconds for orchestration verification.")
    p.add_argument("--bppm_mode", type=str, default=None,
                   choices=["precomputed", "on_the_fly"],
                   help="BPPM mode (forwarded to pretrain)")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="BPPM cache directory for on_the_fly mode (forwarded to pretrain)")
    p.add_argument("--skip_boundary", action="store_true",
                   help="Skip boundary ablation evaluation (forwarded to evaluate)")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader num_workers (forwarded to all stages)")
    p.add_argument("--pin_memory", action="store_true",
                   help="DataLoader pin_memory (forwarded to all stages)")
    p.add_argument("--persistent_workers", action="store_true",
                   help="DataLoader persistent_workers (forwarded to all stages)")
    p.add_argument("--prefetch_workers", type=int, default=None,
                   help="Background threads for async BPPM prefetch in on_the_fly mode (forwarded to pretrain)")
    return p.parse_args()


def main():
    args = parse_args()
    modes = _parse_csv(args.modes)
    seeds = [int(s) for s in _parse_csv(args.seeds)]
    if not modes:
        raise SystemExit("No modes specified.")
    if not seeds:
        raise SystemExit("No seeds specified.")

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.base_dir, "experiments", args.base_name)
    os.makedirs(output_dir, exist_ok=True)

    manifest: Dict = {
        "created_at": timestamp,
        "base_dir": args.base_dir,
        "base_name": args.base_name,
        "modes": modes,
        "seeds": seeds,
        "smoke": bool(args.smoke),
        "output_dir": output_dir,
        "runs": [],
    }

    for mode in modes:
        for seed in seeds:
            print(f"\n=== run mode={mode} seed={seed} ===")
            record = _do_one_run(mode, seed, args)
            manifest["runs"].append(record)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nWrote manifest -> {manifest_path}")

    failed = [r for r in manifest["runs"] if any(
        not r["phases"].get(stage, {}).get("ok", False) for stage in ("pretrain", "finetune", "evaluate")
    )]
    if failed:
        print(f"[WARN] {len(failed)} / {len(manifest['runs'])} runs had a failed stage. See logs in {output_dir}")
    else:
        print(f"All {len(manifest['runs'])} runs completed successfully.")


if __name__ == "__main__":
    main()
