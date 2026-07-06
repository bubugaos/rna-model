"""Parameter counts and FLOPs estimation for the three chunking modes.

Functions:
  - count_params(model, group_by_module=True) -> dict
  - estimate_flops(model, batch_size, seq_len, device='cpu') -> dict

CLI:
    python -m utils.model_stats --batch_size 1 --seq_len 256

prints a side-by-side comparison of no_chunk / fixed / dynamic.

FLOPs are reported by :class:`torch.utils.flop_counter.FlopCounterMode` which
counts dispatcher-level matmul / convolution flops only. It under-counts
elementwise ops (softmax, GELU, LayerNorm) but is consistent across modes so
the ratio between modes is the meaningful comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn

# Make ``src`` importable when this file is run as a script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.model.model import RNADynamicModel  # noqa: E402


def count_params(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """Return parameter counts grouped by top-level submodule, plus a ``total``
    entry. Each entry is a dict with keys ``params`` and ``trainable``.
    """
    per_module: Dict[str, Dict[str, int]] = {}
    total = 0
    trainable = 0

    for name, sub in model.named_children():
        sp = 0
        st = 0
        for p in sub.parameters():
            sp += int(p.numel())
            if p.requires_grad:
                st += int(p.numel())
        per_module[name] = {"params": sp, "trainable": st}
        total += sp
        trainable += st

    # Parameters that live directly on the root module (rare here, but safe).
    direct = 0
    direct_trainable = 0
    for n, p in model.named_parameters(recurse=False):
        direct += int(p.numel())
        if p.requires_grad:
            direct_trainable += int(p.numel())
    if direct > 0:
        per_module["__root__"] = {"params": direct, "trainable": direct_trainable}
        total += direct
        trainable += direct_trainable

    per_module["total"] = {"params": int(total), "trainable": int(trainable)}
    return per_module


def estimate_flops(model: nn.Module, batch_size: int, seq_len: int,
                   vocab_size: int = 7, device: str = "cpu") -> Dict[str, float]:
    """Run a single dummy forward inside ``FlopCounterMode`` and return total
    FLOPs (matmul / conv) plus a per-module breakdown."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as e:  # pragma: no cover - PyTorch < 2.1
        raise RuntimeError(
            "torch.utils.flop_counter is not available. Upgrade PyTorch (>=2.1) "
            "or replace this function with a manual estimator."
        ) from e

    model = model.to(device).eval()
    x = torch.randint(0, max(1, vocab_size - 3), (batch_size, seq_len), device=device, dtype=torch.long)
    row_sum = torch.rand(batch_size, seq_len, device=device)
    entropy = torch.rand(batch_size, seq_len, device=device)
    cross_pair_sum = torch.rand(batch_size, max(seq_len - 1, 0), device=device)
    mask = torch.ones(batch_size, seq_len, device=device)

    flop_counter = FlopCounterMode(display=False, depth=2)
    with flop_counter:
        with torch.no_grad():
            _ = model(x, row_sum=row_sum, entropy=entropy,
                      cross_pair_sum=cross_pair_sum, mask=mask)

    flop_dict = flop_counter.get_flop_counts()
    total_flops = float(flop_dict.get("Global", {}).get("Global", 0))
    if total_flops == 0:
        # Some PyTorch versions expose totals under different keys; sum manually.
        total_flops = 0.0
        for _, ops in flop_dict.items():
            for _, v in ops.items():
                total_flops += float(v)
    return {
        "total_flops": total_flops,
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "device": str(device),
    }


def _build_model(mode: str, embed_dim: int, nhead: int, local_layers: int,
                 latent_layers: int, dim_ff: int, num_classes: int,
                 chunk_size: int = 8) -> nn.Module:
    if mode == "no_chunk":
        kw = dict(use_no_chunk=True)
    elif mode == "fixed":
        kw = dict(use_fixed_router=True, chunk_size=chunk_size)
    elif mode == "dynamic":
        kw = dict()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return RNADynamicModel(
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=local_layers,
        local_num_layers=local_layers,
        latent_num_layers=latent_layers,
        dim_feedforward=dim_ff,
        num_classes=num_classes,
        use_struct_injection=True,
        **kw,
    )


def format_comparison_table(stats: Dict[str, Dict]) -> str:
    """Pretty-print a side-by-side params / FLOPs table for a dict
    ``{mode_name: {'params': int, 'trainable': int, 'flops': float}}``."""
    if not stats:
        return "(no stats collected)"
    modes = list(stats.keys())
    lines = []
    header = f"{'mode':>10s} {'params':>14s} {'trainable':>14s} {'flops':>16s}"
    lines.append(header)
    lines.append("-" * len(header))
    for m in modes:
        s = stats[m]
        params_m = s.get("params", 0)
        trainable_m = s.get("trainable", 0)
        flops = s.get("flops", float("nan"))
        flops_str = f"{flops:>16.3e}" if isinstance(flops, float) else f"{str(flops):>16s}"
        lines.append(f"{m:>10s} {params_m:>14,d} {trainable_m:>14,d} {flops_str}")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Params + FLOPs comparison across chunking modes")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=192)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--local_layers", type=int, default=2)
    p.add_argument("--latent_layers", type=int, default=6)
    p.add_argument("--dim_ff", type=int, default=768)
    p.add_argument("--num_classes", type=int, default=13)
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--smoke", action="store_true",
                   help="Use very small dimensions for a fast smoke run.")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to write the comparison as JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.batch_size = 1
        args.seq_len = 32
        args.embed_dim = 64
        args.nhead = 4
        args.local_layers = 1
        args.latent_layers = 1
        args.dim_ff = 128
        args.num_classes = 3
        args.chunk_size = 4

    stats = {}
    for mode in ("no_chunk", "fixed", "dynamic"):
        model = _build_model(
            mode,
            embed_dim=args.embed_dim,
            nhead=args.nhead,
            local_layers=args.local_layers,
            latent_layers=args.latent_layers,
            dim_ff=args.dim_ff,
            num_classes=args.num_classes,
            chunk_size=args.chunk_size,
        )
        pc = count_params(model)
        params_total = pc["total"]["params"]
        trainable_total = pc["total"]["trainable"]
        try:
            fl = estimate_flops(
                model, args.batch_size, args.seq_len,
                vocab_size=7, device=args.device,
            )
            flops_val = fl["total_flops"]
        except Exception as e:
            flops_val = float("nan")
            print(f"[warn] FLOPs counting failed for mode={mode}: {e}", file=sys.stderr)
        stats[mode] = {
            "params": params_total,
            "trainable": trainable_total,
            "flops": flops_val,
            "per_module": pc,
        }

    table = format_comparison_table(stats)
    print(table)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
