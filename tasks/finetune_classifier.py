import argparse
from datetime import datetime
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig, RNA_BASE_VOCAB_SIZE, resolve_exp_name_with_mode_suffix
from src.data.data_loader import RealRNADataset, collate_fn
from src.model.model import RNADynamicModel


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune classifier head for RNA families")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--fixed_router", action="store_true", help="Override config to use fixed router")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--no_chunk", action="store_true", help="消融实验：无分块模式")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--backbone_lr_scale", type=float, default=None)
    parser.add_argument("--bppm_mode", type=str, default="precomputed", choices=["precomputed", "on_the_fly"])
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--bppm_cache_dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--pin_memory", action="store_true", help="DataLoader pin_memory")
    parser.add_argument("--persistent_workers", action="store_true", help="DataLoader persistent_workers")
    return parser.parse_args()


def compute_macro_f1(preds, labels):
    unique = sorted(set(labels.tolist()) | set(preds.tolist()))
    if not unique:
        return 0.0
    f1s = []
    for cls in unique:
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
    return float(sum(f1s) / len(f1s))


@torch.no_grad()
def evaluate_classifier(model, dataloader, device, max_steps=None, length_buckets=None):
    """Evaluate classifier. If ``length_buckets`` is provided, also returns
    ``per_bucket`` with accuracy / macro_f1 / n per length bucket."""
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
        x, mask, labels, dotbrackets, row_sum, entropy, cross_pair_sum = batch
        x = x.to(device)
        row_sum = row_sum.to(device)
        entropy = entropy.to(device)
        cross_pair_sum = cross_pair_sum.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        outputs = model(x, row_sum=row_sum, entropy=entropy,
                        cross_pair_sum=cross_pair_sum, mask=mask)
        preds = outputs["class_logits"].argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_lengths.append(mask.sum(dim=1).long().cpu())

        total_expected_segments += float(outputs["expected_segments"].sum().item())
        valid_boundary = (mask[:, :-1] * mask[:, 1:]).bool()
        if valid_boundary.any():
            probs = outputs["boundary_probs"][valid_boundary]
            total_boundary_p += float(probs.sum().item())
            total_boundary_n += int(probs.numel())

    if not all_preds:
        return {
            "accuracy": 0.0, "macro_f1": 0.0,
            "expected_segments": 0.0, "boundary_p_mean": 0.0,
            "per_bucket": {},
        }

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    lengths = torch.cat(all_lengths) if all_lengths else torch.tensor([], dtype=torch.long)

    per_bucket = {}
    if length_buckets is not None and len(length_buckets) >= 2:
        # Local import to keep finetune script independent of evaluate_tasks's
        # CLI surface, while sharing the bucketization logic.
        from tasks.evaluate_tasks import _per_bucket_metrics
        per_bucket = _per_bucket_metrics(preds, labels, lengths, list(length_buckets))

    return {
        "accuracy": float((preds == labels).float().mean().item()),
        "macro_f1": compute_macro_f1(preds, labels),
        "expected_segments": total_expected_segments / max(len(labels), 1),
        "boundary_p_mean": total_boundary_p / total_boundary_n if total_boundary_n > 0 else 0.0,
        "per_bucket": per_bucket,
    }


def finetune_classifier(args=None):
    if args is None:
        args = parse_args()

    if args.cache_dir is None and args.bppm_cache_dir is not None:
        args.cache_dir = args.bppm_cache_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        config = ExperimentConfig()
        exp_dir = tempfile.mkdtemp(prefix="rna_smoke_finetune_")
        pretrained_path = os.path.join(exp_dir, "model.pth")
    else:
        temp_config = ExperimentConfig()
        base_dir = args.base_dir if args.base_dir is not None else temp_config.base_dir
        exp_name = args.exp_name if args.exp_name is not None and str(args.exp_name).strip() else temp_config.exp_name
        if args.exp_dir is not None and str(args.exp_dir).strip():
            exp_dir = os.path.abspath(args.exp_dir)
        else:
            exp_dir = os.path.join(base_dir, "experiments", exp_name)
        config_path = os.path.join(exp_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = ExperimentConfig.from_dict(json.load(f))
            config.base_dir = base_dir
            config.exp_name = exp_name
        else:
            config = ExperimentConfig()
            config.base_dir = base_dir
            config.exp_name = exp_name

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

    # If user did not supply --exp_name explicitly but selected an ablation
    # mode, auto-suffix the experiment name so different modes do not share
    # the same exp_dir (and silently overwrite checkpoints under strict=False).
    if args.exp_name is None and not args.smoke:
        config.exp_name = resolve_exp_name_with_mode_suffix(
            config.exp_name,
            no_chunk=bool(args.no_chunk),
            fixed_router=bool(args.fixed_router),
            chunk_size=config.model.chunk_size,
        )
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.head_lr is not None:
        config.training.head_lr = args.head_lr
    if args.backbone_lr is not None:
        config.training.backbone_lr = args.backbone_lr
    if args.backbone_lr_scale is not None:
        config.training.backbone_lr_scale = args.backbone_lr_scale

    if not args.smoke:
        if args.exp_dir is None or not str(args.exp_dir).strip():
            exp_dir = os.path.join(config.base_dir, "experiments", config.exp_name)
        pretrained_path = os.path.join(exp_dir, "model.pth")

    if config.model.use_fixed_router:
        log_filename = "finetune_classifier_result_fixed.txt"
    elif config.model.use_no_chunk:
        log_filename = "finetune_classifier_result_noChunk.txt"
    else:
        log_filename = "finetune_classifier_result.txt"
    log_path = os.path.join(exp_dir, log_filename)

    def log_info(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_info(f"Start Fine-tuning at {start_time_str}")
    log_info(f"Using device: {device}")
    log_info(f"Experiment dir: {exp_dir}" if not args.smoke else f"Smoke mode: using temp dir {exp_dir}")

    if not args.smoke and not os.path.exists(pretrained_path):
        log_info(f"Error: Pretrained model not found at {pretrained_path}")
        return

    _dl_kw = dict(num_workers=max(0, int(args.num_workers)), pin_memory=bool(args.pin_memory),
                  persistent_workers=bool(args.persistent_workers and int(args.num_workers) > 0))

    if args.smoke:
        class _SmokeDataset(Dataset):
            def __init__(self):
                self.lengths = [10, 12, 9, 11]
                self.labels = [0, 1, 0, 1]
                self.dotbrackets = ["." * length for length in self.lengths]

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
                return x, length, self.labels[idx], self.dotbrackets[idx], row_sum, entropy, cross_pair

        train_dataset = _SmokeDataset()
        val_dataset = _SmokeDataset()
        test_dataset = _SmokeDataset()
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, **_dl_kw)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, **_dl_kw)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, **_dl_kw)
        num_classes = 2
    else:
        label_map_path = os.path.join(exp_dir, "label_map.json")
        train_fasta = config.ft_train_fasta
        train_bppm = config.ft_train_bppm
        train_dotbracket = config.ft_train_dotbracket
        val_fasta = config.ft_val_fasta
        val_bppm = config.ft_val_bppm
        val_dotbracket = config.ft_val_dotbracket
        test_fasta = config.ft_test_fasta
        test_bppm = config.ft_test_bppm
        test_dotbracket = config.ft_test_dotbracket

        log_info("Loading datasets...")
        if args.bppm_mode == "precomputed":
            for path in [train_bppm, val_bppm]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing BPPM file in precomputed mode: {path}")

        def _build_dataset(fasta_path, bppm_path, split_name, label_map_path, unknown_policy, save_label_map, dotbracket_path):
            split_cache_dir = None
            if args.bppm_mode == "on_the_fly" and args.cache_dir:
                split_cache_dir = os.path.join(args.cache_dir, split_name)
            dataset_pkl_path = bppm_path if args.bppm_mode == "precomputed" else None
            return RealRNADataset(
                fasta_path=fasta_path,
                pkl_path=dataset_pkl_path,
                label_map_path=label_map_path,
                unknown_family_policy=unknown_policy,
                save_label_map=save_label_map,
                dotbracket_path=dotbracket_path,
                use_masking=False,
                bppm_mode=args.bppm_mode,
                cache_dir=split_cache_dir,
            )

        train_dataset = RealRNADataset(
            fasta_path=train_fasta,
            pkl_path=train_bppm if args.bppm_mode == "precomputed" else None,
            label_map_path=label_map_path,
            unknown_family_policy="add",
            save_label_map=True,
            dotbracket_path=train_dotbracket if os.path.exists(train_dotbracket) else None,
            use_masking=False,
            bppm_mode=args.bppm_mode,
            cache_dir=os.path.join(args.cache_dir, "train") if args.bppm_mode == "on_the_fly" and args.cache_dir else None,
        )
        val_dataset = _build_dataset(val_fasta, val_bppm, "val", label_map_path, "filter", False, val_dotbracket if os.path.exists(val_dotbracket) else None)
        test_dataset = _build_dataset(test_fasta, test_bppm, "test", label_map_path, "filter", False, test_dotbracket if os.path.exists(test_dotbracket) else None)

        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=collate_fn, **_dl_kw)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn, **_dl_kw)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_fn, **_dl_kw)
        num_classes = len(train_dataset.family_to_id)
        log_info(f"Number of classes: {num_classes}")
        log_info(f"Classes: {train_dataset.family_to_id}")

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
        num_classes=num_classes,
        use_fixed_router=config.model.use_fixed_router,
        chunk_size=config.model.chunk_size,
        use_no_chunk=config.model.use_no_chunk,
        use_struct_injection=config.model.use_struct_injection,
        beta=config.model.beta,
        router_bias_init=config.model.router_bias_init,
        router_decay_len=getattr(config.model, "router_decay_len", 400.0),
        max_seq_len=getattr(config.model, "max_seq_len", 2048),
    )

    log_info(f"Loading pretrained weights from {pretrained_path}")
    if args.smoke:
        torch.save(model.state_dict(), pretrained_path)
    state_dict = torch.load(pretrained_path, map_location="cpu")
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"classifier.weight", "classifier.bias"}
    unexpected_missing = [key for key in missing_keys if key not in allowed_missing]
    if unexpected_missing:
        raise RuntimeError(f"Backbone checkpoint is missing keys: {unexpected_missing}")
    if unexpected_keys:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected_keys}")
    model.to(device)

    head_lr = config.training.head_lr
    backbone_lr = config.training.backbone_lr * float(getattr(config.training, "backbone_lr_scale", 1.0))
    log_info(f"Learning Rates - Head: {head_lr}, Backbone: {backbone_lr}")

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    epochs = config.training.num_epochs
    patience = 8
    best_acc = 0.0
    epochs_no_improve = 0
    max_steps = getattr(config.training, "max_steps", None)

    # Learning rate scheduler: warmup + cosine annealing
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    effective_steps_per_epoch = len(train_loader)
    if max_steps is not None:
        effective_steps_per_epoch = min(effective_steps_per_epoch, max_steps)
    total_opt_steps = epochs * max(effective_steps_per_epoch, 1)
    warmup_ratio = float(getattr(config.training, "warmup_ratio", 0.1))
    warmup_ratio = max(0.0, min(1.0, warmup_ratio))
    warmup_steps = int(warmup_ratio * total_opt_steps)
    if total_opt_steps >= 2:
        warmup_steps = max(1, min(warmup_steps, total_opt_steps - 1))
        cosine_steps = max(1, total_opt_steps - warmup_steps)
        scheduler_warmup = LinearLR(optimizer, start_factor=getattr(config.training, "warmup_start_factor", 0.1), end_factor=1.0, total_iters=warmup_steps)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=getattr(config.training, "cosine_eta_min", 0.0))
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
    else:
        scheduler = None

    # Parse length buckets from config
    length_buckets = None
    if config.length_buckets and str(config.length_buckets).strip() != "":
        try:
            length_buckets = [int(p.strip()) for p in str(config.length_buckets).split(",") if p.strip() != ""]
            if len(length_buckets) < 2:
                length_buckets = None
        except ValueError:
            log_info(f"Warning: could not parse length_buckets='{config.length_buckets}', disabling.")
            length_buckets = None

    interrupted = False
    try:
        for epoch in range(epochs):
            model.train()
            total_ce = 0.0
            steps_done = 0

            for step, batch in enumerate(train_loader):
                if max_steps is not None and step >= max_steps:
                    break
                x, mask, labels, dotbrackets, row_sum, entropy, cross_pair_sum = batch
                x = x.to(device)
                row_sum = row_sum.to(device)
                entropy = entropy.to(device)
                cross_pair_sum = cross_pair_sum.to(device)
                mask = mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(x, row_sum=row_sum, entropy=entropy,
                                cross_pair_sum=cross_pair_sum, mask=mask)
                loss = criterion(outputs["class_logits"], labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_ce += float(loss.item())
                steps_done += 1

            train_ce = total_ce / max(steps_done, 1)
            val_metrics = evaluate_classifier(model, val_loader, device, max_steps=max_steps, length_buckets=length_buckets)
            log_info(
                f"Epoch {epoch + 1}/{epochs}, Train CE: {train_ce:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, Val Macro F1: {val_metrics['macro_f1']:.4f}, "
                f"Diag ExpSeg: {val_metrics['expected_segments']:.2f}, Diag Pmean: {val_metrics['boundary_p_mean']:.3f}"
            )

            if val_metrics["accuracy"] > best_acc:
                best_acc = val_metrics["accuracy"]
                epochs_no_improve = 0
                save_path = os.path.join(exp_dir, "finetuned_model.pth")
                torch.save(model.state_dict(), save_path)
                log_info(f"Saved best model with Acc: {best_acc:.4f} to {save_path}")
            else:
                epochs_no_improve += 1
                log_info(f"No improvement for {epochs_no_improve}/{patience} epochs.")
                if epochs_no_improve >= patience:
                    log_info(f"Early stopping triggered after {patience} epochs without improvement.")
                    break

            if args.smoke:
                break
    except KeyboardInterrupt:
        interrupted = True

    if interrupted:
        log_info("Fine-tuning interrupted by user (Ctrl+C).")
    else:
        test_metrics = evaluate_classifier(model, test_loader, device, max_steps=max_steps, length_buckets=length_buckets)
        log_info(
            f"Test Acc: {test_metrics['accuracy']:.4f}, Test Macro F1: {test_metrics['macro_f1']:.4f}, "
            f"Diag ExpSeg: {test_metrics['expected_segments']:.2f}, Diag Pmean: {test_metrics['boundary_p_mean']:.3f}"
        )
        if test_metrics.get("per_bucket"):
            from tasks.evaluate_tasks import format_length_bucket_table
            log_info("Length-stratified Test Metrics:")
            log_info(format_length_bucket_table(test_metrics["per_bucket"]))

    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_info(f"Fine-tuning finished at {end_time_str}")


if __name__ == "__main__":
    finetune_classifier()
