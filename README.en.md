# RNA-MODEL

Dynamic RNA Language Model - A Transformer-Based RNA Sequence Modeling and Structure-Aware Framework

## Project Overview

RNA-MODEL is a deep learning framework for RNA sequence modeling that employs a dynamic chunking mechanism to adaptively learn the optimal number of semantic chunks in RNA sequences. This project supports multiple training modes, including pretraining, fine-tuning, and downstream task evaluation.

## Key Features

- **Dynamic Routing Mechanism**: Adaptively learns the optimal number of chunks
- **Three Operational Modes**:
  - Dynamic Router + Simple Masking (default)
  - Fixed Chunk Router
  - No Chunking Mode
- **Structure Injection**: Supports injection of RNA secondary structure information (BPPM) into the model
- **Multi-task Support**: Language modeling, sequence reconstruction, boundary detection, and classification tasks
- **MDL Loss**: Adaptive chunk learning based on the Minimum Description Length principle

## Environment Setup

### Dependencies

- Python 3.8+
- PyTorch
- ViennaRNA (for BPPM matrix computation)
- See `requirements.txt` for full details

### Installing ViennaRNA

```bash
# Ubuntu/Debian
sudo apt-get install viennarna

# macOS
brew install vienna-rna

# Compile from source
# Visit https://www.tbi.univie.ac.at/RNA/#download
```

## Quick Start

### Data Preparation

1. **Pretraining data**: Prepare RNA sequences in FASTA format
2. **Precompute BPPM matrices (optional)**:
```bash
python -m utils.generate_bppm --fasta_path your_data.fa --output_dir ./bppm_cache
```

### Training the Model

```bash
# Pretraining
python -m src.training.train --config config.yaml

# Single experiment (Dynamic Router + Simple Masking)
python -m src.training.train --exp_name my_experiment

# Ablation experiments
python -m src.training.train --use_no_chunk true      # No chunking mode
python -m src.training.train --use_fixed_router true  # Fixed chunking
```

### Fine-tuning and Evaluation

```bash
# Fine-tune classifier
python -m tasks.finetune_classifier --checkpoint path/to/checkpoint

# Evaluate tasks
python -m tasks.evaluate_tasks --checkpoint path/to/checkpoint
```

## Project Structure

```
RNA-MODEL/
├── src/
│   ├── config.py          # Configuration management
│   ├── data/
│   │   ├── data_loader.py # Data loading and BPPM preprocessing
│   │   └── masking_utils.py # Masking strategies
│   ├── model/
│   │   ├── model.py       # Main model: RNADynamicModel
│   │   ├── chunking.py    # Chunking and de-chunking modules
│   │   ├── decoder.py     # Decoder
│   │   ├── dynamic_router.py # Dynamic router
│   │   ├── latent_transformer.py # Latent-space Transformer
│   │   ├── local_encoder.py    # Local encoder
│   │   └── positional_encoding.py # Positional encoding
│   └── training/
│       ├── train.py       # Training logic
│       └── losses.py      # Loss functions
├── tasks/
│   ├── evaluate_tasks.py  # Downstream evaluation tasks
│   └── finetune_classifier.py # Classifier fine-tuning
├── utils/
│   ├── aggregate_results.py   # Result aggregation and analysis
│   ├── generate_bppm.py      # BPPM matrix generation
│   ├── generate_family_dotbrackets.py # Structure generation
│   ├── model_stats.py        # Model statistics
│   └── stats_utils.py        # Statistical utilities
├── scripts/
│   └── run_multiseed.py      # Multi-seed experiment script
├── data/                     # Data directory
└── tests/                    # Unit tests
```

## Core Configuration

Model behavior is controlled via `ModelConfig` in `src/config.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_no_chunk` | Enable no-chunking mode | `false` |
| `use_fixed_router` | Use fixed chunk router | `false` |
| `chunk_size` | Fixed chunk size | `8` |
| `beta` | Router temperature parameter | `1.0` |
| `use_struct_injection` | Inject structural features | `true` |

## Loss Functions

- **MLM Loss**: Masked Language Modeling
- **Reconstruction Loss**: Sequence reconstruction
- **MDL Loss**: Adaptive chunking based on Minimum Description Length

## Utility Scripts

### Multi-Seed Experiments

```bash
python scripts/run_multiseed.py --mode dynamic_router --seeds 42 43 44
```

### Model Statistics

```bash
python -m utils.model_stats --mode dynamic_router --embed-dim 256
```

### Result Aggregation

```bash
python -m utils.aggregate_results --manifest manifest.yaml --metrics loss f1
```

## Testing

```bash
# Run all tests
pytest tests/

# Quick smoke test
pytest tests/ -v -k smoke
```

## License

This project is intended solely for academic research purposes.