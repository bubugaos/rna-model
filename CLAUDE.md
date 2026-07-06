# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: When you need to understand the project structure, file responsibilities, or architecture before reading code, start with `document.md` — it maps every source file's purpose, data flow, and modification boundaries. This avoids reading full source files to understand context, saving significant token overhead.

## Environment

Activate the conda environment before running any code:

```bash
conda activate hnet
```

## Dependencies

All Python dependencies are listed in `requirements.txt` (project root). Install with:

```bash
pip install -r requirements.txt
```

ViennaRNA requires conda: `conda install -c bioconda viennarna`

**Rule**: Whenever a new dependency or Python package is added to the project, update `requirements.txt` immediately.

## Common Commands

### Training

```bash
# Single experiment (default: dynamic router with simple masking)
python -m src.training.train

# Ablation modes
python -m src.training.train --no_chunk                          # no chunking baseline
python -m src.training.train --fixed_router --chunk_size 8       # fixed-size chunks

# Key CLI overrides
python -m src.training.train --exp_name my_exp --seed 42 --device cuda \
    --batch_size 64 --lr 8e-5 --num_epochs 300 \
    --fasta_path /path/to/data.fasta --pkl_path /path/to/bppm.pkl

# MDL loss tuning (no target_chunk_ratio needed — model learns chunk count autonomously)
python -m src.training.train --lambda_mdl 0.5 --mdl_cost_base 0.05
```

### Multi-seed pipeline (pretrain -> finetune -> evaluate)

```bash
# Smoke test (fast orchestration check)
python scripts/run_multiseed.py --smoke --seeds 42,43

# Full run
python scripts/run_multiseed.py \
    --base_dir /path/to/RNA-model \
    --base_name rfam_run \
    --modes no_chunk,fixed,dynamic \
    --seeds 42,43,44 \
    --pretrain_epochs 300 --finetune_epochs 300
```

### Finetuning & Evaluation

```bash
python tasks/finetune_classifier.py --exp_name my_exp --base_dir /path/to/RNA-model
python tasks/evaluate_tasks.py --exp_name my_exp --base_dir /path/to/RNA-model
```

### Testing

```bash
# Central smoke test (covers forward/backward, STE gradient, shape consistency)
python tests/test_smoke_pipeline.py

# Run with pytest
pytest tests/
```

### Utilities

```bash
# Precompute BPPM matrices from FASTA (requires ViennaRNA)
python utils/generate_bppm.py

# Precompute dot-bracket structures for boundary evaluation
python utils/generate_family_dotbrackets.py

# Compare model capacity across modes
python -m utils.model_stats

# Aggregate multi-seed results with bootstrap CI + permutation tests
python utils/aggregate_results.py
```

## Architecture Overview

This project implements a **dynamic chunking RNA language model**. It processes RNA sequences (A/U/C/G) through a shared backbone that supports three ablation modes differing only in how tokens are grouped into chunks.

### Three modes (controlled by `use_no_chunk` / `use_fixed_router` in `ModelConfig`)

| Mode | Behavior | CLI flag |
|------|----------|----------|
| `dynamic` | Learned boundary prediction via `DynamicRouter` | (default) |
| `fixed` | Uniform chunks every N tokens (`FixedChunkRouter`) | `--fixed_router` |
| `no_chunk` | Token-level only, no chunking | `--no_chunk` |

All three modes share the same `LocalEncoder + BPPM injection + LatentTransformer + ReconDecoder + classifier`. Only the presence of `Router -> Downsampler -> Dechunker` differs.

### Data Flow

```
RNA sequence (A/U/C/G)
  -> LocalEncoder (token embeddings + positional encoding + transformer)
  -> BPPM structural injection (pairing probability + entropy features)
  -> [Router -> Downsampler] (token -> chunk; skipped in no_chunk mode)
  -> LatentTransformer (chunk-level or token-level transformer)
  -> [Dechunker] (chunk -> token with EMA smoothing; skipped in no_chunk mode)
  -> ReconDecoder -> reconstruction logits (output vocab)
  -> Classifier (mean pool over valid positions -> class logits)
```

### Key Components

- **`src/config.py`**: Single source of truth for all config. `ExperimentConfig` nests `MaskingConfig`, `ModelConfig`, `LossConfig`, `TrainingConfig`. CLI args override defaults at runtime.
- **`src/model/dynamic_router.py`**: Core chunking logic. `DynamicRouter` fuses cosine similarity (semantic boundary) with BPPM cross-pairing scores (structural boundary) into `boundary_probs` via sigmoid. Uses STE for gradient flow through hard boundary decisions.
- **`src/model/chunking.py`**: `Downsampler` (first token per segment) and `Dechunker` (EMA-smoothed upsample back to token-level with residual connection).
- **`src/training/losses.py`**: `total_loss = lambda_recon * recon_loss + lambda_mlm * mlm_loss + lambda_mdl * mdl_weight * mdl_loss`. Reconstruction loss on unmasked positions; MLM on masked positions; MDL loss penalizes boundaries whose adjacent chunk representations are too similar (cosine distance < structure-aware cost), letting the model learn chunk count autonomously without `target_chunk_ratio`.
- **`src/data/data_loader.py`**: `RealRNADataset` loads FASTA + BPPM (precomputed PKL or ViennaRNA on-the-fly with caching). Supports family labels and dot-bracket structures. `collate_fn` handles variable-length padding.
- **`src/data/masking_utils.py`**: Two masking strategies — `SimpleMasking` (BERT-style random) and `PairingAwareMasking` (uses BPPM to jointly mask paired bases).
- **`scripts/run_multiseed.py`**: Drives the full pretrain -> finetune -> evaluate pipeline across mode x seed combinations, writing `manifest.json` for aggregation.
- **`tasks/finetune_classifier.py`**: Fine-tunes the classifier head with optional backbone fine-tuning at reduced LR. Supports length-bucketed evaluation.
- **`tasks/evaluate_tasks.py`**: Evaluates classification accuracy + boundary prediction quality (P/R/F1 at tolerances across truth modes).

### ViennaRNA Dependency

BPPM computation requires the ViennaRNA Python bindings (`import RNA`). Precompute offline with `utils/generate_bppm.py`, or use `--bppm_mode on_the_fly` with optional `--cache_dir` for caching.

### Configuration is the single source of truth

`src/config.py` defines all hyperparameters. CLI arguments in training/evaluation scripts override these defaults. When adding new hyperparameters, add them to the relevant dataclass in `config.py` first, then expose via CLI in the scripts that need them. Exp names auto-suffix with mode (`_noChunk`, `_fixed_cs8`) when using ablation flags, preventing accidental checkpoint overwrites between modes.
