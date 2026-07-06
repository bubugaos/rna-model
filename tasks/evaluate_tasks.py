import torch
from torch.utils.data import DataLoader
import sys
import os
import json
import argparse
from torch.utils.data import Dataset

try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None

# Ensure src module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.model import RNADynamicModel
from src.data.data_loader import RealRNADataset, collate_fn
from src.config import ExperimentConfig, resolve_exp_name_with_mode_suffix

BOUNDARY_TRUTH_MODES = ("all_transitions", "stem_endpoints", "stem_loop_junctions")


def dotbracket_to_boundary(dotbracket, mode='all_transitions'):
    """Convert a dot-bracket secondary structure into a set of boundary indices.

    A "boundary at index i" means an edge between position i and position i+1.

    Modes:
      - 'all_transitions':       every adjacent character change is a boundary.
                                 (Loose proxy; matches the original behavior.)
      - 'stem_endpoints':        only transitions between paired ('(', ')') and
                                 unpaired ('.') characters. Captures stem-loop
                                 boundaries (helix start / helix end).
      - 'stem_loop_junctions':   only paired-to-paired transitions where the
                                 orientation flips: '(' -> ')' (immediate
                                 hairpin close) or ')' -> '(' (back-to-back
                                 helices). Sparse but biologically meaningful.
    """
    boundaries = []
    if not dotbracket:
        return boundaries
    if mode not in BOUNDARY_TRUTH_MODES:
        raise ValueError(
            f"Unknown boundary truth mode: {mode}. Must be one of {BOUNDARY_TRUTH_MODES}."
        )

    paired = {'(', ')'}
    for i in range(len(dotbracket) - 1):
        c1 = dotbracket[i]
        c2 = dotbracket[i + 1]
        if c1 == c2:
            continue
        if mode == 'all_transitions':
            boundaries.append(i)
        elif mode == 'stem_endpoints':
            # exactly one of the two characters is paired
            if (c1 in paired) ^ (c2 in paired):
                boundaries.append(i)
        elif mode == 'stem_loop_junctions':
            # both paired, orientation flips
            if (c1 in paired) and (c2 in paired) and (c1 != c2):
                boundaries.append(i)
    return boundaries


def compute_boundary_metrics(pred_mask_or_bounds, true_bounds, seq_len, tolerance=1):
    """
    pred_mask_or_bounds: either a 1-D tensor of {0,1} with shape (L-1,) OR a
                         list/array of integer boundary indices.
    true_bounds:         list of integer boundary indices.
    tolerance:           absolute distance |pred - true| considered a match.

    Returns precision, recall, f1 (each in [0, 1]).
    """
    if isinstance(pred_mask_or_bounds, torch.Tensor):
        pred_bounds = torch.where(pred_mask_or_bounds == 1)[0].tolist()
    else:
        pred_bounds = list(pred_mask_or_bounds)

    pred_bounds = [int(p) for p in pred_bounds if 0 <= int(p) < seq_len - 1]
    true_bounds = [int(t) for t in true_bounds if 0 <= int(t) < seq_len - 1]

    if not true_bounds and not pred_bounds:
        return 1.0, 1.0, 1.0
    if not true_bounds and pred_bounds:
        return 0.0, 0.0, 0.0
    if true_bounds and not pred_bounds:
        return 0.0, 0.0, 0.0

    tp = 0
    matched_true = set()
    for p in pred_bounds:
        for t in true_bounds:
            if t not in matched_true and abs(p - t) <= tolerance:
                tp += 1
                matched_true.add(t)
                break
    precision = tp / len(pred_bounds) if pred_bounds else 0.0

    tp_recall = 0
    matched_pred = set()
    for t in true_bounds:
        for p in pred_bounds:
            if p not in matched_pred and abs(p - t) <= tolerance:
                tp_recall += 1
                matched_pred.add(p)
                break
    recall = tp_recall / len(true_bounds) if true_bounds else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

DEFAULT_LENGTH_BUCKETS = (100, 150, 250, 400)


def _bucket_label(length, buckets):
    """Return the bucket-label string for ``length`` given a sorted list of
    boundary values like (100, 150, 250, 400) -> buckets [100,150), [150,250),
    [250,400]. The final bucket is inclusive on the upper bound. Returns None
    if ``length`` falls outside the configured range."""
    if not buckets or length is None:
        return None
    if length < buckets[0]:
        return None
    if length > buckets[-1]:
        return None
    for i in range(len(buckets) - 1):
        lo = buckets[i]
        hi = buckets[i + 1]
        if i == len(buckets) - 2:
            if lo <= length <= hi:
                return f"[{lo},{hi}]"
        else:
            if lo <= length < hi:
                return f"[{lo},{hi})"
    return None


def _macro_f1_from_tensors(preds, labels):
    """Macro-averaged F1 over the union of labels in preds and labels.
    Falls back to a pure-pytorch implementation if scikit-learn is not
    available."""
    if preds.numel() == 0:
        return 0.0
    if f1_score is not None:
        return float(f1_score(labels.numpy(), preds.numpy(), average='macro'))
    unique = sorted(set(labels.tolist()) | set(preds.tolist()))
    f1s = []
    for c in unique:
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def _per_bucket_metrics(preds, labels, lengths, buckets):
    """Compute accuracy + macro_f1 + count per length bucket.
    Returns dict keyed by bucket label, plus an 'out_of_range' bucket for
    samples whose length is outside the configured range."""
    if buckets is None or len(buckets) < 2:
        return {}
    bucket_labels = []
    for L in lengths.tolist():
        bucket_labels.append(_bucket_label(int(L), buckets) or "out_of_range")
    metrics = {}
    unique_buckets = sorted(set(bucket_labels))
    for bl in unique_buckets:
        idx = [i for i, x in enumerate(bucket_labels) if x == bl]
        if not idx:
            continue
        sel = torch.tensor(idx, dtype=torch.long)
        p_sel = preds[sel]
        l_sel = labels[sel]
        acc = (p_sel == l_sel).float().mean().item() if p_sel.numel() else 0.0
        metrics[bl] = {
            "accuracy": float(acc),
            "macro_f1": _macro_f1_from_tensors(p_sel, l_sel),
            "n": int(p_sel.numel()),
        }
    return metrics


def format_length_bucket_table(per_bucket):
    if not per_bucket:
        return "(length buckets: no samples)"
    lines = []
    header = f"{'bucket':>14s} {'accuracy':>10s} {'macro_f1':>10s} {'n':>6s}"
    lines.append(header)
    lines.append("-" * len(header))
    for bl in sorted(per_bucket.keys()):
        r = per_bucket[bl]
        lines.append(f"{bl:>14s} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['n']:>6d}")
    return "\n".join(lines)


@torch.no_grad()
def evaluate_classification(model, dataloader, device, max_steps=None, length_buckets=None):
    """Classification evaluation. If ``length_buckets`` is provided (an
    iterable of sorted ints, e.g. ``(100, 150, 250, 400)``), per-bucket
    accuracy / macro_f1 / sample count are also returned under the
    ``per_bucket`` key."""
    model.eval()
    all_preds = []
    all_labels = []
    all_lengths = []
    total_expected_segments = 0.0
    total_boundary_p = 0.0
    total_boundary_n = 0

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        if len(batch) == 10:
            x, raw_x, mask, mask_pos, mlm_labels, cls_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch
        else:
            x, mask, cls_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch

        x = x.to(device)
        row_sum = row_sum.to(device)
        entropy = entropy.to(device)
        cross_pair_sum = cross_pair_sum.to(device)
        mask = mask.to(device)
        cls_labels = cls_labels.to(device)

        outputs = model(x, row_sum=row_sum, entropy=entropy,
                        cross_pair_sum=cross_pair_sum, mask=mask)
        logits = outputs["class_logits"]
        preds = logits.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(cls_labels.cpu())
        all_lengths.append(mask.sum(dim=1).long().cpu())

        total_expected_segments += float(outputs["expected_segments"].sum().item())
        valid_boundary = (mask[:, :-1] * mask[:, 1:]).bool()
        if valid_boundary.any():
            probs = outputs["boundary_probs"][valid_boundary]
            total_boundary_p += float(probs.sum().item())
            total_boundary_n += int(probs.numel())

    if not all_preds:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "expected_segments": 0.0,
            "boundary_p_mean": 0.0,
            "per_bucket": {},
        }

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_lengths = torch.cat(all_lengths) if all_lengths else torch.tensor([], dtype=torch.long)

    accuracy = (all_preds == all_labels).float().mean().item()
    f1_macro = _macro_f1_from_tensors(all_preds, all_labels)

    per_bucket = {}
    if length_buckets is not None and len(length_buckets) >= 2:
        per_bucket = _per_bucket_metrics(all_preds, all_labels, all_lengths, list(length_buckets))

    return {
        "accuracy": accuracy,
        "macro_f1": f1_macro,
        "expected_segments": total_expected_segments / max(len(all_labels), 1),
        "boundary_p_mean": total_boundary_p / total_boundary_n if total_boundary_n > 0 else 0.0,
        "per_bucket": per_bucket,
    }

@torch.no_grad()
def evaluate_boundary(model, dataloader, device, max_steps=None,
                     truth_mode='all_transitions', tolerance=1):
    """Single (truth_mode, tolerance) boundary evaluation.

    Kept for backward compatibility. Returns (precision, recall, f1).
    For an ablation over multiple modes/tolerances use
    :func:`evaluate_boundary_ablation` instead.
    """
    model.eval()
    precisions, recalls, f1s = [], [], []
    has_dotbracket = False

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        if len(batch) == 10:
            x, raw_x, mask, mask_pos, mlm_labels, cls_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch
        else:
            x, mask, cls_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch

        x = x.to(device)
        row_sum = row_sum.to(device)
        entropy = entropy.to(device)
        cross_pair_sum = cross_pair_sum.to(device)
        mask = mask.to(device)

        outputs = model(x, row_sum=row_sum, entropy=entropy,
                        cross_pair_sum=cross_pair_sum, mask=mask)
        boundary_mask = outputs["boundary_mask"]

        for j in range(x.size(0)):
            if dotbrackets[j] is None:
                continue
            has_dotbracket = True
            seq_len = int(mask[j].sum().item())
            if seq_len <= 1:
                continue

            pred = boundary_mask[j, :seq_len - 1]
            true_bounds = dotbracket_to_boundary(dotbrackets[j], mode=truth_mode)
            p, r, f = compute_boundary_metrics(pred.cpu(), true_bounds, seq_len, tolerance=tolerance)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

    if not has_dotbracket:
        print("Warning: No dotbracket data provided, boundary evaluation skipped.")
        return 0.0, 0.0, 0.0

    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls) / len(recalls) if recalls else 0
    avg_f = sum(f1s) / len(f1s) if f1s else 0
    return avg_p, avg_r, avg_f


@torch.no_grad()
def evaluate_boundary_ablation(model, dataloader, device, tolerances, truth_modes,
                               max_steps=None):
    """Compute P/R/F1 for every (tolerance, truth_mode) combination in one
    pass over the dataloader. Returns a nested dict:

        results[truth_mode][tolerance] = {
            'precision': float, 'recall': float, 'f1': float, 'n': int
        }

    If the dataloader contains no samples with dot-bracket labels, returns an
    empty dict.
    """
    model.eval()
    # Collect per-sample (pred_indices, seq_len, dotbracket) triples once,
    # then iterate (mode x tol) cheaply on Python side.
    samples = []
    has_dotbracket = False

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        if len(batch) == 10:
            x, raw_x, mask, mask_pos, mlm_labels, cls_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch
        else:
            x, mask, cls_labels, dotbrackets, \
                row_sum, entropy, cross_pair_sum = batch

        x = x.to(device)
        row_sum = row_sum.to(device)
        entropy = entropy.to(device)
        cross_pair_sum = cross_pair_sum.to(device)
        mask = mask.to(device)

        outputs = model(x, row_sum=row_sum, entropy=entropy,
                        cross_pair_sum=cross_pair_sum, mask=mask)
        boundary_mask = outputs["boundary_mask"]

        for j in range(x.size(0)):
            db = dotbrackets[j]
            if db is None:
                continue
            has_dotbracket = True
            seq_len = int(mask[j].sum().item())
            if seq_len <= 1:
                continue
            pred_row = boundary_mask[j, :seq_len - 1].detach().cpu()
            pred_bounds = torch.where(pred_row >= 0.5)[0].tolist()
            samples.append((pred_bounds, seq_len, db))

    if not has_dotbracket:
        print("Warning: No dotbracket data provided, boundary ablation skipped.")
        return {}

    results = {}
    for mode in truth_modes:
        results[mode] = {}
        precomputed_truth = [dotbracket_to_boundary(db, mode=mode) for (_, _, db) in samples]
        for tol in tolerances:
            ps, rs, fs = [], [], []
            for (pred_bounds, seq_len, _), true_bounds in zip(samples, precomputed_truth):
                p, r, f = compute_boundary_metrics(pred_bounds, true_bounds, seq_len, tolerance=tol)
                ps.append(p)
                rs.append(r)
                fs.append(f)
            results[mode][int(tol)] = {
                "precision": float(sum(ps) / len(ps)) if ps else 0.0,
                "recall": float(sum(rs) / len(rs)) if rs else 0.0,
                "f1": float(sum(fs) / len(fs)) if fs else 0.0,
                "n": int(len(ps)),
            }
    return results


def format_boundary_ablation_table(results):
    """Return a human-readable text table for a results dict produced by
    :func:`evaluate_boundary_ablation`. Empty dict -> a single-line message."""
    if not results:
        return "(boundary ablation: no dot-bracket labels available)"
    lines = []
    header = f"{'truth_mode':22s} {'tol':>4s} {'precision':>11s} {'recall':>8s} {'f1':>6s} {'n':>5s}"
    lines.append(header)
    lines.append("-" * len(header))
    for mode in sorted(results.keys()):
        for tol in sorted(results[mode].keys()):
            r = results[mode][tol]
            lines.append(
                f"{mode:22s} {tol:>4d} {r['precision']:>11.4f} {r['recall']:>8.4f} {r['f1']:>6.4f} {r['n']:>5d}"
            )
    return "\n".join(lines)

def _parse_int_list(text):
    if text is None or str(text).strip() == "":
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _parse_str_list(text, allowed=None):
    if text is None or str(text).strip() == "":
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip() != ""]
    if allowed is not None:
        for p in parts:
            if p not in allowed:
                raise ValueError(f"Unknown value '{p}'. Allowed: {sorted(allowed)}")
    return parts


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate downstream classification with boundary diagnostics")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--fixed_router", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument('--no_chunk', action='store_true',
                    help='消融实验：无分块模式')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--bppm_mode", type=str, default="precomputed", choices=["precomputed", "on_the_fly"])
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--bppm_cache_dir", type=str, default=None)
    parser.add_argument(
        "--boundary_tolerances", type=str, default="0,1,2,3",
        help="Comma-separated tolerance values (in tokens) for the boundary ablation. Default: '0,1,2,3'."
    )
    parser.add_argument(
        "--boundary_truth_modes", type=str,
        default=",".join(BOUNDARY_TRUTH_MODES),
        help=(
            "Comma-separated truth-mode definitions for dot-bracket -> boundary "
            "conversion. Allowed: all_transitions, stem_endpoints, stem_loop_junctions."
        )
    )
    parser.add_argument(
        "--skip_boundary", action="store_true",
        help="Skip boundary ablation evaluation entirely. Useful when dot-bracket files are not available."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--pin_memory", action="store_true", help="DataLoader pin_memory")
    parser.add_argument("--persistent_workers", action="store_true", help="DataLoader persistent_workers")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.cache_dir is None and args.bppm_cache_dir is not None:
        args.cache_dir = args.bppm_cache_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Boundary definition: adjacent character change in dot-bracket (dotbracket[i] != dotbracket[i+1]).")
    
    # 1. Load Configuration
    # Determine base_dir and exp_name early to find config file
    temp_config = ExperimentConfig()
    base_dir = args.base_dir if args.base_dir is not None else temp_config.base_dir

    exp_name_provided = args.exp_name is not None and str(args.exp_name).strip() != ""
    if exp_name_provided:
        exp_name = args.exp_name
    else:
        # Auto-suffix default exp_name with the chunking mode so the evaluate
        # command points at the right ablation experiment dir without having
        # to explicitly pass --exp_name every time.
        exp_name = resolve_exp_name_with_mode_suffix(
            temp_config.exp_name,
            no_chunk=bool(args.no_chunk),
            fixed_router=bool(args.fixed_router),
            chunk_size=args.chunk_size,
        )

    if args.exp_dir is not None and str(args.exp_dir).strip():
        exp_dir = os.path.abspath(args.exp_dir)
    else:
        exp_dir = os.path.join(base_dir, "experiments", exp_name)
    config_path = os.path.join(exp_dir, "config.json")
    
    # Load config from file or use defaults
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ExperimentConfig.from_dict(config_dict)
        # Ensure base_dir is correct (might differ from saved config)
        config.base_dir = base_dir
    else:
        print("No config file found, using default configuration.")
        config = ExperimentConfig()
        config.base_dir = base_dir
        config.exp_name = exp_name
    
    # 2. Override with CLI arguments
    if args.base_dir is not None:
        config.base_dir = args.base_dir
    if args.exp_name is not None:
        config.exp_name = args.exp_name
        
    if args.fixed_router:
        config.model.use_fixed_router = True
    if args.chunk_size is not None:
        config.model.chunk_size = args.chunk_size
    if args.no_chunk:
        config.model.use_no_chunk = True
        
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps

    # Determine log file name
    if config.model.use_fixed_router:
        log_filename = "evaluate_tasks_result_fixed.txt"
    elif config.model.use_no_chunk:
        log_filename = "evaluate_tasks_result_noChunk.txt"
    else:
        log_filename = "evaluate_tasks_result.txt"
        
    log_path = os.path.join(exp_dir, log_filename)
    
    def log_info(msg):
        print(msg)
        # Only log to file if not smoke test, or ensure dir exists
        if not args.smoke and os.path.exists(exp_dir):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    # Parse ablation knobs (shared between smoke and real flows)
    tolerances = _parse_int_list(args.boundary_tolerances) or [1]
    truth_modes = _parse_str_list(args.boundary_truth_modes, allowed=set(BOUNDARY_TRUTH_MODES)) or list(BOUNDARY_TRUTH_MODES)
    length_buckets = _parse_int_list(config.length_buckets)
    if length_buckets is not None and len(length_buckets) < 2:
        length_buckets = None

    # Smoke Test Mode
    if args.smoke:
        class _SmokeDataset(Dataset):
            def __init__(self):
                # Lengths span the default length buckets to exercise the
                # stratification code path.
                self.lengths = [10, 12, 9, 11, 110, 165, 280]
                self.labels = [0, 1, 0, 1, 0, 1, 0]
                self.dotbrackets = [
                    "....((((..",
                    "....))))....",
                    "." * 9,
                    "..((..))...",
                    "." * 110,
                    "((....))" + "." * 157,
                    "(((...)))" + "." * 271,
                ]

            def __len__(self):
                return len(self.lengths)

            def __getitem__(self, idx):
                import numpy as np
                length = self.lengths[idx]
                x = torch.randint(0, 4, (length,), dtype=torch.long)
                # Generate fake BPPM vectors matching the new interface
                row_sum = np.random.rand(length).astype(np.float32)
                entropy = np.random.rand(length).astype(np.float32)
                cross_pair = np.random.rand(max(length - 1, 0)).astype(np.float32)
                label = self.labels[idx]
                dotbracket = self.dotbrackets[idx]
                return x, length, label, dotbracket, row_sum, entropy, cross_pair

        test_dataset = _SmokeDataset()
        _dl_kw = dict(num_workers=max(0, int(args.num_workers)), pin_memory=bool(args.pin_memory),
                      persistent_workers=bool(args.persistent_workers and int(args.num_workers) > 0))
        test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn, **_dl_kw)
        num_classes = 2

        model = RNADynamicModel(
            input_vocab_size=config.model.input_vocab_size,
            output_vocab_size=config.model.output_vocab_size,
            embed_dim=config.model.embed_dim,
            nhead=config.model.nhead,
            num_layers=config.model.num_layers,
            local_num_layers=config.model.local_num_layers,
            latent_num_layers=config.model.latent_num_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            beta=config.model.beta,
            num_classes=num_classes,
            use_fixed_router=config.model.use_fixed_router,
            chunk_size=config.model.chunk_size,
            use_no_chunk=config.model.use_no_chunk,
            use_struct_injection=config.model.use_struct_injection,
            router_bias_init=config.model.router_bias_init,
            router_decay_len=getattr(config.model, "router_decay_len", 400.0),
            max_seq_len=getattr(config.model, "max_seq_len", 2048),
        )
        model.to(device)

        log_info("\nEvaluating Downstream Classification...")
        cls_metrics = evaluate_classification(
            model, test_loader, device,
            max_steps=config.training.max_steps,
            length_buckets=length_buckets,
        )
        log_info(f"Classification Accuracy: {cls_metrics['accuracy']:.4f}")
        log_info(f"Classification Macro F1: {cls_metrics['macro_f1']:.4f}")
        log_info(f"Chunk Diagnostics - Expected Segments: {cls_metrics['expected_segments']:.2f}")
        log_info(f"Chunk Diagnostics - Boundary P Mean: {cls_metrics['boundary_p_mean']:.3f}")
        if cls_metrics.get("per_bucket"):
            log_info("\nLength-stratified Classification:")
            log_info(format_length_bucket_table(cls_metrics["per_bucket"]))

        if args.skip_boundary:
            log_info("\nBoundary Ablation: skipped (--skip_boundary).")
            boundary_results = {"skipped": True}
        else:
            log_info("\nBoundary Ablation (tolerance x truth_mode):")
            boundary_results = evaluate_boundary_ablation(
                model, test_loader, device,
                tolerances=tolerances,
                truth_modes=truth_modes,
                max_steps=config.training.max_steps,
            )
            log_info(format_boundary_ablation_table(boundary_results))
        return

    # Real Execution
    # Model Path
    finetuned_path = os.path.join(exp_dir, 'finetuned_model.pth')
    model_path = finetuned_path if os.path.exists(finetuned_path) else os.path.join(exp_dir, 'model.pth')
    log_info(f"Loading model from {model_path}")
    
    # Paths (from config — each can be a file or shard directory)
    train_fasta = config.ft_train_fasta
    train_bppm = config.ft_train_bppm
    test_fasta = config.ft_test_fasta
    test_bppm = config.ft_test_bppm
    test_dotbracket = config.ft_test_dotbracket
    
    if args.bppm_mode == "precomputed":
        for p in [train_bppm, test_bppm]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing BPPM file in precomputed mode: {p}")

    def _build_dataset(fasta_path, bppm_path, split_name, label_map_path, dotbracket_path):
        split_cache_dir = None
        if args.bppm_mode == "on_the_fly" and args.cache_dir:
            split_cache_dir = os.path.join(args.cache_dir, split_name)
        dataset_pkl_path = bppm_path if args.bppm_mode == "precomputed" else None
        return RealRNADataset(
            fasta_path=fasta_path,
            pkl_path=dataset_pkl_path,
            label_map_path=label_map_path,
            dotbracket_path=dotbracket_path,
            use_masking=False,
            bppm_mode=args.bppm_mode,
            cache_dir=split_cache_dir
        )

    # Generate/Load Label Map from Train
    map_path = os.path.join(exp_dir, 'label_map.json')
    train_map_ready = os.path.exists(train_fasta) and (args.bppm_mode != "precomputed" or os.path.exists(train_bppm))
    if os.path.exists(map_path):
        log_info(f"Using existing label map: {map_path}")
    elif train_map_ready:
        log_info("label_map.json missing; reconstructing from train split.")
        temp_train = _build_dataset(
            fasta_path=train_fasta,
            bppm_path=train_bppm,
            split_name="train",
            label_map_path=None,
            dotbracket_path=None
        )
        with open(map_path, 'w') as f:
            json.dump(temp_train.family_to_id, f)
    else:
        raise FileNotFoundError(
            f"Missing label_map.json at {map_path}, and train split is unavailable to reconstruct it."
        )
    
    # Load Test Dataset
    log_info("Loading test dataset...")
    test_dataset = _build_dataset(
        fasta_path=test_fasta,
        bppm_path=test_bppm,
        split_name="test",
        label_map_path=map_path,
        dotbracket_path=test_dotbracket if os.path.exists(test_dotbracket) else None,
    )
    
    _dl_kw = dict(num_workers=max(0, int(args.num_workers)), pin_memory=bool(args.pin_memory),
                  persistent_workers=bool(args.persistent_workers and int(args.num_workers) > 0))
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, collate_fn=collate_fn, **_dl_kw)
    num_classes = len(test_dataset.family_to_id)
    log_info(f"Num classes: {num_classes}")
    
    model = RNADynamicModel(
        input_vocab_size=config.model.input_vocab_size,
        output_vocab_size=config.model.output_vocab_size,
        embed_dim=config.model.embed_dim, 
        nhead=config.model.nhead, 
        num_layers=config.model.num_layers,
        local_num_layers=config.model.local_num_layers,
        latent_num_layers=config.model.latent_num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        beta=config.model.beta,
        num_classes=num_classes,
        use_fixed_router=config.model.use_fixed_router,
        chunk_size=config.model.chunk_size,
        use_no_chunk=config.model.use_no_chunk,
        use_struct_injection=config.model.use_struct_injection,
        router_bias_init=config.model.router_bias_init,
        router_decay_len=getattr(config.model, "router_decay_len", 400.0),
        max_seq_len=getattr(config.model, "max_seq_len", 2048),
    )
    
    state_dict = torch.load(model_path, map_location=device)
    # Filter out classifier if size mismatch (though here we expect match if fine-tuned)
    model_dict = model.state_dict()
    if 'classifier.weight' in state_dict:
        if state_dict['classifier.weight'].shape != model_dict['classifier.weight'].shape:
            log_info("Warning: Classifier shape mismatch. Skipping classifier weights.")
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"classifier.weight", "classifier.bias"}
    unexpected_missing = [key for key in missing_keys if key not in allowed_missing]
    if unexpected_missing:
        raise RuntimeError(f"Checkpoint is missing non-classifier keys: {unexpected_missing}")
    if unexpected_keys:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected_keys}")
    model.to(device)
    
    log_info("\nEvaluating Downstream Classification...")
    cls_metrics = evaluate_classification(
        model, test_loader, device,
        max_steps=config.training.max_steps,
        length_buckets=length_buckets,
    )
    log_info(f"Classification Accuracy: {cls_metrics['accuracy']:.4f}")
    log_info(f"Classification Macro F1: {cls_metrics['macro_f1']:.4f}")
    log_info(f"Chunk Diagnostics - Expected Segments: {cls_metrics['expected_segments']:.2f}")
    log_info(f"Chunk Diagnostics - Boundary P Mean: {cls_metrics['boundary_p_mean']:.3f}")
    if cls_metrics.get("per_bucket"):
        log_info("\nLength-stratified Classification:")
        log_info(format_length_bucket_table(cls_metrics["per_bucket"]))

    if args.skip_boundary:
        log_info("\nBoundary Ablation: skipped (--skip_boundary).")
        boundary_results = {"skipped": True}
    else:
        log_info("\nBoundary Ablation (tolerance x truth_mode):")
        boundary_results = evaluate_boundary_ablation(
            model, test_loader, device,
            tolerances=tolerances,
            truth_modes=truth_modes,
            max_steps=config.training.max_steps,
        )
        log_info(format_boundary_ablation_table(boundary_results))

    # Persist machine-readable evaluation summary alongside the human log so
    # downstream aggregation (utils/aggregate_results.py) can pick it up
    # without re-parsing free-form text.
    eval_summary = {
        "exp_name": config.exp_name,
        "use_no_chunk": bool(config.model.use_no_chunk),
        "use_fixed_router": bool(config.model.use_fixed_router),
        "chunk_size": int(config.model.chunk_size),
        "num_classes": int(num_classes),
        "classification": {
            "accuracy": cls_metrics["accuracy"],
            "macro_f1": cls_metrics["macro_f1"],
            "expected_segments": cls_metrics["expected_segments"],
            "boundary_p_mean": cls_metrics["boundary_p_mean"],
            "per_bucket": cls_metrics.get("per_bucket", {}),
        },
        "boundary_ablation": boundary_results,
        "boundary_tolerances": tolerances,
        "boundary_truth_modes": truth_modes,
        "length_buckets": length_buckets,
    }
    summary_filename = log_filename.replace(".txt", "_summary.json")
    summary_path = os.path.join(exp_dir, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    log_info(f"Wrote evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
