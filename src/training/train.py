import argparse
from datetime import datetime
import inspect
import json
import os
import random
import sys
from urllib.parse import urlparse

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.config import ExperimentConfig, RNA_BASE_VOCAB_SIZE, RNA_TOKEN_IDS, resolve_exp_name_with_mode_suffix
from src.data.data_loader import RealRNADataset, collate_fn
from src.model.model import RNADynamicModel
from src.training.losses import RNALMCriterion

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def apply_tau(model, tau=None):
    model_ref = unwrap_model(model)
    if tau is not None and hasattr(model_ref, "dynamic_router") and hasattr(model_ref.dynamic_router, "set_tau"):
        tau_value = float(tau)
        current_tau = getattr(model_ref.dynamic_router, "tau", None)
        if current_tau is None or float(current_tau) != tau_value:
            model_ref.dynamic_router.set_tau(tau_value)


def parse_args():
    parser = argparse.ArgumentParser(description="RNA Dynamic Model Training")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--fixed_router", action="store_true", help="Use fixed chunk router")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size for fixed chunk router")
    parser.add_argument("--no_chunk", action="store_true", help="消融实验：禁用分块，使用纯LocalEncoder+线性重构头")

    parser.add_argument("--use_masking", type=lambda x: (str(x).lower() == "true"), default=True, help="Enable MLM (true/false)")
    parser.add_argument("--masking_type", type=str, default=None, choices=["simple", "pairing_aware"], help="Masking type")

    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment output directory (absolute or relative)")

    parser.add_argument("--bppm_mode", type=str, default="precomputed", choices=["precomputed", "on_the_fly"], help="BPPM loading mode for RealRNADataset")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional BPPM cache directory for on_the_fly mode")
    parser.add_argument("--bppm_cache_dir", type=str, default=None, help="Deprecated alias of --cache_dir")

    parser.add_argument("--smoke", action="store_true", help="Run a minimal smoke path with synthetic data")
    parser.add_argument("--max_steps", type=int, default=None, help="Max batches per epoch (train/eval)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers for train/val")
    parser.add_argument("--pin_memory", action="store_true", help="Enable DataLoader pin_memory for train/val")
    parser.add_argument("--persistent_workers", action="store_true", help="Enable DataLoader persistent_workers for train/val")
    parser.add_argument("--prefetch_workers", type=int, default=0, help="Background threads for async BPPM prefetch (on_the_fly mode)")

    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio over total steps")
    parser.add_argument("--warmup_start_factor", type=float, default=None, help="Linear warmup start factor")
    parser.add_argument("--cosine_eta_min", type=float, default=None, help="Cosine LR scheduler eta_min")

    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="DDP backend (nccl/gloo)")
    parser.add_argument("--dist_url", type=str, default="env://", help="DDP init method")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP launcher")
    parser.add_argument("--rank", type=int, default=0, help="Global rank fallback for DDP")
    parser.add_argument("--world_size", type=int, default=1, help="World size fallback for DDP")
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed(args):
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", str(args.rank)))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))

    use_ddp = bool(args.ddp) or env_world_size > 1
    rank = env_rank
    world_size = env_world_size if env_world_size > 1 else int(args.world_size)
    local_rank = env_local_rank

    if not use_ddp:
        return False, 0, 1, -1
    if world_size < 2:
        print("[WARN] --ddp enabled but world_size < 2; fallback to single process.")
        return False, 0, 1, -1
    if local_rank < 0:
        local_rank = rank if torch.cuda.is_available() else -1
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    dist_url = args.dist_url
    if os.name == "nt" and str(args.dist_backend).lower() == "gloo":
        parsed = urlparse(dist_url)
        if parsed.scheme == "env" and "use_libuv=" not in parsed.query:
            sep = "&" if "?" in dist_url else "?"
            dist_url = f"{dist_url}{sep}use_libuv=0"
            if rank == 0:
                print(f"[DDP] Windows gloo compatibility: set dist_url={dist_url}")

    if not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend, init_method=dist_url, rank=rank, world_size=world_size)
    return True, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def _mlm_debug_stats(output, mask, mask_pos, mlm_labels):
    stats = {
        "mlm_n": 0,
        "mlm_acc": float("nan"),
        "masked_ce": float("nan"),
        "unmasked_ce": float("nan"),
        "masked_logits_std": float("nan"),
        "masked_logits_mean": float("nan"),
        "label_max": float("nan"),
        "label_min": float("nan"),
        "num_classes": float("nan"),
    }
    if mask_pos is None or mlm_labels is None:
        return stats

    if mask is None:
        valid_pos = torch.ones_like(mask_pos, dtype=torch.bool)
    else:
        valid_pos = mask.bool()

    recon_logits = output["recon_logits"]
    stats["num_classes"] = int(recon_logits.size(-1))

    masked_sel = mask_pos & valid_pos
    unmasked_sel = (~mask_pos) & valid_pos

    if masked_sel.any():
        masked_logits = recon_logits[masked_sel]
        masked_labels = mlm_labels[masked_sel]
        stats["mlm_n"] = int(masked_labels.numel())
        stats["label_max"] = int(masked_labels.max().item())
        stats["label_min"] = int(masked_labels.min().item())
        stats["masked_logits_std"] = float(masked_logits.detach().std().item())
        stats["masked_logits_mean"] = float(masked_logits.detach().mean().item())
        pred = masked_logits.argmax(dim=-1)
        stats["mlm_acc"] = float((pred == masked_labels).float().mean().item())
        stats["masked_ce"] = float(torch.nn.functional.cross_entropy(masked_logits, masked_labels, reduction="mean").item())

    if unmasked_sel.any():
        unmasked_logits = recon_logits[unmasked_sel]
        unmasked_labels = mlm_labels[unmasked_sel]
        stats["unmasked_ce"] = float(torch.nn.functional.cross_entropy(unmasked_logits, unmasked_labels, reduction="mean").item())

    return stats


def _router_epoch_stats(output, mask):
    boundary_probs = output.get("boundary_probs")
    if boundary_probs is None:
        boundary_probs = output["boundary_mask"]

    if mask is None:
        valid_boundary = torch.ones_like(boundary_probs, dtype=torch.bool)
    else:
        valid_boundary = (mask[:, :-1] * mask[:, 1:]).bool()

    exp_seg = 1.0 + (boundary_probs * valid_boundary.to(dtype=boundary_probs.dtype)).sum(dim=1)
    hard_seg = 1 + ((output["boundary_mask"] > 0.5) & valid_boundary).sum(dim=1)

    if not valid_boundary.any():
        return {
            "expected_segments": exp_seg,
            "hard_segments": hard_seg,
            "p_mean": float("nan"),
            "p_min": float("nan"),
            "p_max": float("nan"),
            "valid_n": 0,
        }

    p_valid = boundary_probs[valid_boundary]
    return {
        "expected_segments": exp_seg,
        "hard_segments": hard_seg,
        "p_mean": float(p_valid.mean().item()),
        "p_min": float(p_valid.min().item()),
        "p_max": float(p_valid.max().item()),
        "valid_n": int(p_valid.numel()),
    }


def build_rna_lm_criterion(config: ExperimentConfig) -> RNALMCriterion:
    criterion_signature = inspect.signature(RNALMCriterion.__init__)
    criterion_kwargs = {"vocab_size": config.model.output_vocab_size}
    for param_name in criterion_signature.parameters:
        if param_name in {"self", "vocab_size"}:
            continue
        if hasattr(config.loss, param_name):
            criterion_kwargs[param_name] = getattr(config.loss, param_name)
    return RNALMCriterion(**criterion_kwargs)


def train_one_epoch(model, dataloader, criterion, optimizer, device, tau, epoch_idx, max_grad_norm=1.0, use_mlm=False, scheduler=None, max_steps=None, debug_mlm=False, grad_accum_steps=1):
    model.train()
    apply_tau(model, tau=tau)

    total_metrics = {"total_loss": 0.0, "recon_loss": 0.0, "mlm_loss": 0.0, "mdl_loss": 0.0}
    accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad()
    steps = len(dataloader) if max_steps is None else min(len(dataloader), max_steps)

    router_samples = 0
    router_exp_seg_sum = 0.0
    router_hard_seg_sum = 0.0
    router_p_sum = 0.0
    router_p_n = 0
    router_p_min = float("inf")
    router_p_max = float("-inf")
    steps_done = 0

    for i, batch in enumerate(dataloader):
        if max_steps is not None and i >= max_steps:
            break
        steps_done += 1

        if use_mlm:
            x, raw_x, mask, mask_pos, mlm_labels, family_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch
            x, raw_x = x.to(device), raw_x.to(device)
            mask = mask.to(device)
            mask_pos = mask_pos.to(device)
            mlm_labels = mlm_labels.to(device)
            row_sum = row_sum.to(device)
            entropy = entropy.to(device)
            cross_pair_sum = cross_pair_sum.to(device)
            model_input = x
            target_x = raw_x
        else:
            x, mask, family_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch
            x, mask = x.to(device), mask.to(device)
            row_sum = row_sum.to(device)
            entropy = entropy.to(device)
            cross_pair_sum = cross_pair_sum.to(device)
            model_input = x
            target_x = x
            mask_pos, mlm_labels = None, None

        output = model(model_input, row_sum=row_sum, entropy=entropy,
                       cross_pair_sum=cross_pair_sum, mask=mask)
        loss_dict = criterion(output, target_x, row_sum, mask, mask_pos, mlm_labels)
        loss = loss_dict["total_loss"]
        (loss / accum_steps).backward()

        should_step = ((i + 1) % accum_steps == 0) or ((i + 1) == steps)
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        for k in total_metrics:
            if k in loss_dict:
                total_metrics[k] += float(loss_dict[k].item())

        with torch.no_grad():
            r = _router_epoch_stats(output, mask)
            router_samples += x.size(0)
            router_exp_seg_sum += float(r["expected_segments"].sum().item())
            router_hard_seg_sum += float(r["hard_segments"].sum().item())
            if r["valid_n"] > 0 and not np.isnan(r["p_mean"]):
                router_p_sum += r["p_mean"] * r["valid_n"]
                router_p_n += r["valid_n"]
                router_p_min = min(router_p_min, r["p_min"])
                router_p_max = max(router_p_max, r["p_max"])

        if (i + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            dbg = {}
            if debug_mlm and use_mlm:
                with torch.no_grad():
                    dbg = _mlm_debug_stats(output, mask, mask_pos, mlm_labels)
            print(
                f"Epoch {epoch_idx + 1}, Step {i + 1}/{steps}, "
                f"Loss: {loss_dict['total_loss'].item():.4f}, "
                f"Recon: {loss_dict.get('recon_loss', torch.tensor(0.0)).item():.4f}, "
                f"MLM: {loss_dict.get('mlm_loss', torch.tensor(0.0)).item():.4f}, "
                f"MDL: {loss_dict.get('mdl_loss', torch.tensor(0.0)).item():.4f}, "
                f"LR: {lr:.2e}"
                + (
                    f", MLM_n: {dbg.get('mlm_n', 0)}, MLM_acc: {dbg.get('mlm_acc', float('nan')):.3f}, "
                    f"masked_ce: {dbg.get('masked_ce', float('nan')):.4f}, "
                    f"unmasked_ce: {dbg.get('unmasked_ce', float('nan')):.4f}, "
                    f"logits_std: {dbg.get('masked_logits_std', float('nan')):.3f}"
                    if debug_mlm and use_mlm else ""
                )
            )

    ddp_ready = dist.is_available() and dist.is_initialized()
    stat_device = next(model.parameters()).device
    metric_keys = list(total_metrics.keys())
    metric_sums = torch.tensor([float(total_metrics[k]) for k in metric_keys], dtype=torch.float64, device=stat_device)
    agg_vals = torch.tensor(
        [
            float(steps_done),
            float(router_samples),
            float(router_exp_seg_sum),
            float(router_hard_seg_sum),
            float(router_p_sum),
            float(router_p_n),
        ],
        dtype=torch.float64,
        device=stat_device,
    )
    router_min_t = torch.tensor(float(router_p_min), dtype=torch.float64, device=stat_device)
    router_max_t = torch.tensor(float(router_p_max), dtype=torch.float64, device=stat_device)

    if ddp_ready:
        dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(agg_vals, op=dist.ReduceOp.SUM)
        dist.all_reduce(router_min_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(router_max_t, op=dist.ReduceOp.MAX)

    for idx, k in enumerate(metric_keys):
        total_metrics[k] = float(metric_sums[idx].item())

    global_steps = int(agg_vals[0].item())
    router_samples = float(agg_vals[1].item())
    router_exp_seg_sum = float(agg_vals[2].item())
    router_hard_seg_sum = float(agg_vals[3].item())
    router_p_sum = float(agg_vals[4].item())
    router_p_n = int(agg_vals[5].item())
    router_p_min = float(router_min_t.item())
    router_p_max = float(router_max_t.item())

    denom = global_steps if global_steps > 0 else 1
    for k in total_metrics:
        total_metrics[k] /= denom

    total_metrics["expected_segments"] = router_exp_seg_sum / router_samples if router_samples > 0 else 0.0
    total_metrics["hard_segments"] = router_hard_seg_sum / router_samples if router_samples > 0 else 0.0
    if router_p_n > 0:
        total_metrics["boundary_p_mean"] = router_p_sum / router_p_n
        total_metrics["boundary_p_min"] = router_p_min
        total_metrics["boundary_p_max"] = router_p_max
    else:
        total_metrics["boundary_p_mean"] = float("nan")
        total_metrics["boundary_p_min"] = float("nan")
        total_metrics["boundary_p_max"] = float("nan")
    return total_metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_mlm=False, max_steps=None, debug_mlm=False, result_path=None):
    model.eval()
    total_metrics = {"total_loss": 0.0, "recon_loss": 0.0, "mlm_loss": 0.0, "mdl_loss": 0.0}
    total_chunks = 0.0
    total_samples = 0.0
    router_samples = 0
    router_exp_seg_sum = 0.0
    router_hard_seg_sum = 0.0
    router_p_sum = 0.0
    router_p_n = 0
    router_p_min = float("inf")
    router_p_max = float("-inf")
    steps_done = 0
    steps = len(dataloader) if max_steps is None else min(len(dataloader), max_steps)

    for i, batch in enumerate(dataloader):
        if max_steps is not None and i >= max_steps:
            break
        steps_done += 1
        if use_mlm:
            x, raw_x, mask, mask_pos, mlm_labels, family_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch
            x, raw_x = x.to(device), raw_x.to(device)
            mask = mask.to(device)
            mask_pos = mask_pos.to(device)
            mlm_labels = mlm_labels.to(device)
            row_sum = row_sum.to(device)
            entropy = entropy.to(device)
            cross_pair_sum = cross_pair_sum.to(device)
            model_input = x
            target_x = raw_x
        else:
            x, mask, family_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch
            x, mask = x.to(device), mask.to(device)
            row_sum = row_sum.to(device)
            entropy = entropy.to(device)
            cross_pair_sum = cross_pair_sum.to(device)
            model_input = x
            target_x = x
            mask_pos, mlm_labels = None, None

        output = model(model_input, row_sum=row_sum, entropy=entropy,
                       cross_pair_sum=cross_pair_sum, mask=mask)
        loss_dict = criterion(output, target_x, row_sum, mask, mask_pos, mlm_labels)

        for k in total_metrics:
            if k in loss_dict:
                total_metrics[k] += float(loss_dict[k].item())

        num_cuts = output["boundary_mask"].sum(dim=-1)
        total_chunks += float((num_cuts + 1).sum().item())
        total_samples += float(x.size(0))

        r = _router_epoch_stats(output, mask)
        router_samples += x.size(0)
        router_exp_seg_sum += float(r["expected_segments"].sum().item())
        router_hard_seg_sum += float(r["hard_segments"].sum().item())
        if r["valid_n"] > 0 and not np.isnan(r["p_mean"]):
            router_p_sum += r["p_mean"] * r["valid_n"]
            router_p_n += r["valid_n"]
            router_p_min = min(router_p_min, r["p_min"])
            router_p_max = max(router_p_max, r["p_max"])

        if debug_mlm and use_mlm and i == 0:
            dbg = _mlm_debug_stats(output, mask, mask_pos, mlm_labels)
            msg = (
                f"[Eval MLM dbg] MLM_n: {dbg.get('mlm_n', 0)}, MLM_acc: {dbg.get('mlm_acc', float('nan')):.3f}, "
                f"masked_ce: {dbg.get('masked_ce', float('nan')):.4f}, "
                f"unmasked_ce: {dbg.get('unmasked_ce', float('nan')):.4f}, "
                f"label_range: [{dbg.get('label_min', float('nan'))}, {dbg.get('label_max', float('nan'))}], "
                f"num_classes: {dbg.get('num_classes', float('nan'))}"
            )
            print(msg)
            if result_path:
                with open(result_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

    ddp_ready = dist.is_available() and dist.is_initialized()
    stat_device = next(model.parameters()).device
    metric_keys = list(total_metrics.keys())
    metric_sums = torch.tensor([float(total_metrics[k]) for k in metric_keys], dtype=torch.float64, device=stat_device)
    agg_vals = torch.tensor(
        [
            float(steps_done),
            float(total_chunks),
            float(total_samples),
            float(router_samples),
            float(router_exp_seg_sum),
            float(router_hard_seg_sum),
            float(router_p_sum),
            float(router_p_n),
        ],
        dtype=torch.float64,
        device=stat_device,
    )
    router_min_t = torch.tensor(float(router_p_min), dtype=torch.float64, device=stat_device)
    router_max_t = torch.tensor(float(router_p_max), dtype=torch.float64, device=stat_device)

    if ddp_ready:
        dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(agg_vals, op=dist.ReduceOp.SUM)
        dist.all_reduce(router_min_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(router_max_t, op=dist.ReduceOp.MAX)

    for idx, k in enumerate(metric_keys):
        total_metrics[k] = float(metric_sums[idx].item())

    global_steps = int(agg_vals[0].item())
    total_chunks = float(agg_vals[1].item())
    total_samples = float(agg_vals[2].item())
    router_samples = float(agg_vals[3].item())
    router_exp_seg_sum = float(agg_vals[4].item())
    router_hard_seg_sum = float(agg_vals[5].item())
    router_p_sum = float(agg_vals[6].item())
    router_p_n = int(agg_vals[7].item())
    router_p_min = float(router_min_t.item())
    router_p_max = float(router_max_t.item())

    denom = global_steps if global_steps > 0 else 1
    for k in total_metrics:
        total_metrics[k] /= denom

    avg_chunks = total_chunks / total_samples if total_samples > 0 else 0.0
    total_metrics["expected_segments"] = router_exp_seg_sum / router_samples if router_samples > 0 else 0.0
    total_metrics["hard_segments"] = router_hard_seg_sum / router_samples if router_samples > 0 else 0.0
    if router_p_n > 0:
        total_metrics["boundary_p_mean"] = router_p_sum / router_p_n
        total_metrics["boundary_p_min"] = router_p_min
        total_metrics["boundary_p_max"] = router_p_max
    else:
        total_metrics["boundary_p_mean"] = float("nan")
        total_metrics["boundary_p_min"] = float("nan")
        total_metrics["boundary_p_max"] = float("nan")
    return total_metrics, avg_chunks


def main():
    args = parse_args()
    use_ddp, rank, world_size, local_rank = setup_distributed(args)

    try:
        set_seed(args.seed)

        if args.smoke:
            class _SmokeDataset(Dataset):
                def __init__(self, use_masking: bool):
                    self.use_masking = use_masking
                    self.lengths = [12, 9, 15, 10]
                    self.labels = [0, 1, 0, 1]
                    self.dotbrackets = ["." * l for l in self.lengths]

                def __len__(self):
                    return len(self.lengths)

                def __getitem__(self, idx):
                    import numpy as np
                    length = self.lengths[idx]
                    x = torch.randint(0, RNA_BASE_VOCAB_SIZE, (length,), dtype=torch.long)
                    # Generate fake BPPM vectors matching the new interface
                    row_sum = np.random.rand(length).astype(np.float32)
                    entropy = np.random.rand(length).astype(np.float32)
                    cross_pair = np.random.rand(max(length - 1, 0)).astype(np.float32)
                    family_label = self.labels[idx]
                    dotbracket = self.dotbrackets[idx]
                    if self.use_masking:
                        raw_x = x.clone()
                        masked_x = x.clone()
                        mask_pos = torch.zeros(length, dtype=torch.bool)
                        mask_pos[0] = True
                        masked_x[0] = RNA_TOKEN_IDS["MASK"]
                        labels = raw_x.clone()
                        return masked_x, raw_x, length, mask_pos, labels, family_label, dotbracket, row_sum, entropy, cross_pair
                    return x, length, family_label, dotbracket, row_sum, entropy, cross_pair

        config = ExperimentConfig()
        if args.use_masking is not None:
            config.masking.use_masking = args.use_masking
        if args.masking_type is not None:
            config.masking.masking_type = args.masking_type
        if args.batch_size is not None:
            config.training.batch_size = args.batch_size
        if args.num_epochs is not None:
            config.training.num_epochs = args.num_epochs
        if args.lr is not None:
            config.training.lr = args.lr
        if use_ddp:
            if os.name == "nt" and str(args.dist_backend).lower() == "gloo":
                config.training.device = "cpu"
            elif torch.cuda.is_available():
                if local_rank < 0:
                    raise ValueError("DDP with CUDA requires valid LOCAL_RANK/local_rank.")
                torch.cuda.set_device(local_rank)
                config.training.device = f"cuda:{local_rank}"
            else:
                config.training.device = "cpu"

        if args.cache_dir is None and args.bppm_cache_dir is not None:
            args.cache_dir = args.bppm_cache_dir
        if args.warmup_ratio is not None:
            config.training.warmup_ratio = args.warmup_ratio
        if args.warmup_start_factor is not None:
            config.training.warmup_start_factor = args.warmup_start_factor
        if args.cosine_eta_min is not None:
            config.training.cosine_eta_min = args.cosine_eta_min
        if args.fixed_router:
            config.model.use_fixed_router = True
        if args.no_chunk:
            config.model.use_no_chunk = True
        if args.chunk_size is not None:
            config.model.chunk_size = args.chunk_size

        # Exp name: if the user explicitly passed --exp_name we respect it
        # verbatim. If they did NOT, but enabled an ablation flag
        # (--no_chunk / --fixed_router), we auto-suffix to avoid silently
        # sharing the same exp_dir between modes and overwriting checkpoints.
        if args.exp_name:
            config.exp_name = args.exp_name
        else:
            config.exp_name = resolve_exp_name_with_mode_suffix(
                config.exp_name,
                no_chunk=bool(args.no_chunk),
                fixed_router=bool(args.fixed_router),
                chunk_size=config.model.chunk_size,
            )

        print(f"=== Experiment: {config.exp_name} ===")
        print(f"Device: {config.training.device}")
        print(f"MLM Mode: {config.masking.use_masking} ({config.masking.masking_type})")
        print(f"DDP: {use_ddp} (rank={rank}, world_size={world_size}, local_rank={local_rank})")

        exp_dir = os.path.abspath(args.exp_dir) if args.exp_dir is not None else os.path.abspath(os.path.join(config.base_dir, "experiments", config.exp_name))
        os.makedirs(exp_dir, exist_ok=True)

        if is_main_process(rank):
            config.save(os.path.join(exp_dir, "config.json"))
            with open(os.path.join(exp_dir, "run_info.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "seed": args.seed,
                        "rank": rank,
                        "world_size": world_size,
                        "ddp": use_ddp,
                        "bppm_mode": args.bppm_mode,
                        "cache_dir": args.cache_dir,
                        "device": config.training.device,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        if args.smoke:
            full_dataset = _SmokeDataset(use_masking=config.masking.use_masking)
        elif not os.path.exists(config.fasta_path):
            print(f"Error: FASTA file not found at {config.fasta_path}")
            return
        elif args.bppm_mode == "precomputed" and not os.path.exists(config.pkl_path):
            print(f"Error: PKL file not found at {config.pkl_path} (required for bppm_mode=precomputed)")
            return
        else:
            dataset_kwargs = dict(use_masking=config.masking.use_masking, masking_type=config.masking.masking_type, masking_config=config.masking)
            dataset_sig = inspect.signature(RealRNADataset.__init__)
            if "bppm_mode" in dataset_sig.parameters:
                dataset_kwargs["bppm_mode"] = args.bppm_mode
            if "cache_dir" in dataset_sig.parameters:
                dataset_kwargs["cache_dir"] = args.cache_dir
            if "prefetch_workers" in dataset_sig.parameters:
                dataset_kwargs["prefetch_workers"] = args.prefetch_workers
            dataset_pkl_path = config.pkl_path if args.bppm_mode == "precomputed" else None
            full_dataset = RealRNADataset(config.fasta_path, dataset_pkl_path, **dataset_kwargs)
        
        # NOTE: BPPM prefetch (if enabled) continues in background without blocking.
        # Cache misses during training are handled by synchronous fallback in __getitem__.
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        split_generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=split_generator)

        if is_main_process(rank):
            split_path = os.path.join(exp_dir, "split.json")
            split_payload = {
                "seed": args.seed,
                "train_size": train_size,
                "val_size": val_size,
                "train_indices": [int(i) for i in train_dataset.indices],
                "val_indices": [int(i) for i in val_dataset.indices],
            }
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(split_payload, f, ensure_ascii=False)

        print(f"Dataset: {len(full_dataset)} total, {len(train_dataset)} train, {len(val_dataset)} val")

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_ddp else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers and int(args.num_workers) > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers and int(args.num_workers) > 0),
        )

        local_num_layers = config.model.local_num_layers
        latent_num_layers = config.model.latent_num_layers
        shared_num_layers = config.model.num_layers

        model = RNADynamicModel(
            input_vocab_size=config.model.input_vocab_size,
            output_vocab_size=config.model.output_vocab_size,
            embed_dim=config.model.embed_dim,
            nhead=config.model.nhead,
            num_layers=shared_num_layers,
            local_num_layers=local_num_layers,
            latent_num_layers=latent_num_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            beta=config.model.beta,
            router_bias_init=config.model.router_bias_init,
            router_decay_len=config.model.router_decay_len,
            use_fixed_router=config.model.use_fixed_router,
            chunk_size=config.model.chunk_size,
            use_no_chunk=config.model.use_no_chunk,
            use_struct_injection=config.model.use_struct_injection,
            max_seq_len=config.model.max_seq_len,
        ).to(config.training.device)

        if use_ddp:
            use_cuda_ddp = str(config.training.device).startswith("cuda")
            ddp_device_ids = [local_rank] if use_cuda_ddp and local_rank >= 0 else None
            model = DDP(model, device_ids=ddp_device_ids)

        criterion = build_rna_lm_criterion(config)
        optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

        effective_steps_per_epoch = len(train_loader)
        if args.max_steps is not None:
            effective_steps_per_epoch = min(effective_steps_per_epoch, args.max_steps)
        grad_accum_steps = max(1, int(config.training.grad_accum_steps))
        optimizer_updates_per_epoch = (max(0, int(effective_steps_per_epoch)) + grad_accum_steps - 1) // grad_accum_steps
        total_steps = config.training.num_epochs * optimizer_updates_per_epoch
        warmup_ratio = float(config.training.warmup_ratio)
        warmup_ratio = max(0.0, min(1.0, warmup_ratio))
        warmup_steps = int(warmup_ratio * total_steps)
        if total_steps < 2:
            print(f"[WARN] total_steps={total_steps} too small for warmup+cosine scheduler; scheduler disabled.")
            scheduler = None
        else:
            warmup_steps = max(1, warmup_steps)
            warmup_steps = min(warmup_steps, total_steps - 1)
            cosine_steps = max(1, total_steps - warmup_steps)
            scheduler1 = LinearLR(optimizer, start_factor=config.training.warmup_start_factor, end_factor=1.0, total_iters=warmup_steps)
            scheduler2 = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=config.training.cosine_eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

        history = {
            "train_total": [],
            "train_recon": [],
            "train_mlm": [],
            "train_mdl": [],
            "val_total": [],
            "val_recon": [],
            "val_mlm": [],
            "val_mdl": [],
            "lr": [],
        }
        best_val_loss = float("inf")
        best_model_path = os.path.join(exp_dir, "model.pth")
        early_stop_patience = 8
        epochs_since_improve = 0
        print("Starting training...")

        result_path = os.path.join(exp_dir, "train_result.txt")
        if is_main_process(rank):
            with open(result_path, "w", encoding="utf-8") as f:
                start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Start Training Experiment: {config.exp_name} at {start_time_str}\n")

        model_for_save = unwrap_model(model)
        interrupted = False
        try:
            for epoch in range(config.training.num_epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                ratio = epoch / max(config.training.num_epochs - 1, 1)
                tau = config.training.tau_init - (config.training.tau_init - config.training.tau_final) * ratio
                apply_tau(model_for_save, tau=tau)

                warmup_epochs = max(1, int(config.loss.mdl_warmup_ratio * config.training.num_epochs))
                mdl_w = min(1.0, epoch / warmup_epochs)
                criterion.set_mdl_weight(mdl_w)

                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    config.training.device,
                    tau,
                    epoch,
                    max_grad_norm=config.training.grad_clip,
                    use_mlm=config.masking.use_masking,
                    scheduler=scheduler,
                    max_steps=args.max_steps,
                    debug_mlm=False,
                    grad_accum_steps=config.training.grad_accum_steps,
                )
                val_metrics, avg_chunks = evaluate(
                    model,
                    val_loader,
                    criterion,
                    config.training.device,
                    use_mlm=config.masking.use_masking,
                    max_steps=args.max_steps,
                    debug_mlm=False,
                    result_path=result_path,
                )

                for k in ["total", "recon", "mlm", "mdl"]:
                    history[f"train_{k}"].append(train_metrics.get(f"{k}_loss", 0.0))
                    history[f"val_{k}"].append(val_metrics.get(f"{k}_loss", 0.0))
                history["lr"].append(optimizer.param_groups[0]["lr"])

                msg = (
                    f"Epoch [{epoch + 1}/{config.training.num_epochs}] - "
                    f"Train Loss: {train_metrics['total_loss']:.4f} - "
                    f"Val Loss: {val_metrics['total_loss']:.4f} - "
                    f"Recon: {train_metrics.get('recon_loss', 0.0):.4f} - "
                    f"MLM: {train_metrics.get('mlm_loss', 0.0):.4f} - "
                    f"MDL: {train_metrics.get('mdl_loss', 0.0):.4f} - "
                    f"AvgChunks: {avg_chunks:.2f} - "
                    f"ExpSeg(train/val): {train_metrics.get('expected_segments', 0.0):.2f}/{val_metrics.get('expected_segments', 0.0):.2f} - "
                    f"Pmean(train/val): {train_metrics.get('boundary_p_mean', float('nan')):.3f}/{val_metrics.get('boundary_p_mean', float('nan')):.3f}"
                )
                if is_main_process(rank):
                    print(msg)
                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")

                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    if is_main_process(rank):
                        print(f"have save best model with val loss {best_val_loss:.4f}")
                        torch.save(model_for_save.state_dict(), best_model_path)
                        config.save(os.path.join(exp_dir, "config.json"))
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= early_stop_patience:
                        msg = f"Early stopping: no improvement for {early_stop_patience} epochs."
                        if is_main_process(rank):
                            print(msg)
                            with open(result_path, "a", encoding="utf-8") as f:
                                f.write(msg + "\n")
                        break
                if args.smoke:
                    break
        except KeyboardInterrupt:
            interrupted = True

        if is_main_process(rank):
            with open(os.path.join(exp_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f)
            end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(result_path, "a", encoding="utf-8") as f:
                if interrupted:
                    f.write("Training interrupted by user (Ctrl+C).\n")
                f.write(f"Training finished at {end_time_str}\n")
            if interrupted:
                print(f"Training interrupted. Results saved to {exp_dir}")
            else:
                print(f"Training finished. Results saved to {exp_dir}")
    finally:
        if use_ddp and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
