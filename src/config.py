from dataclasses import dataclass, field, fields
from typing import Optional, List, Dict, Any
import os
import json

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RNA_TOKEN_IDS: Dict[str, int] = {
    "A": 0,
    "U": 1,
    "C": 2,
    "G": 3,
    "PAD": 4,
    "MASK": 5,
    "UNK": 6,
}
RNA_BASE_VOCAB_SIZE = 4
RNA_INPUT_VOCAB_SIZE = 7
RNA_OUTPUT_VOCAB_SIZE = 7

@dataclass
class MaskingConfig:
    """Masking strategy configuration"""
    use_masking: bool = True
    masking_type: str = 'pairing_aware'  # 'simple' or 'pairing_aware'
    mask_prob: float = 0.13
    coupled_prob: float = 0.5
    pairing_threshold: float = 0.3
    mask_token_id: int = RNA_TOKEN_IDS["MASK"]
    replace_prob: float = 0.8
    random_prob: float = 0.1

@dataclass
class ModelConfig:
    """Model configuration"""
    input_vocab_size: int = RNA_INPUT_VOCAB_SIZE
    output_vocab_size: int = RNA_OUTPUT_VOCAB_SIZE
    embed_dim: int = 384
    nhead: int = 6
    # Backward-compatible shared depth. Kept for old configs/checkpoints.
    num_layers: int = 4
    # Preferred explicit depths for two transformer stages.
    local_num_layers: Optional[int] = 2
    latent_num_layers: Optional[int] = 2
    dim_feedforward: int = 384
    dropout: float = 0.1
    use_fixed_router: bool = False
    chunk_size: int = 3
    use_struct_injection: bool = True
    beta: float = 1     # DynamicRouter 的结构权重
    router_bias_init: float = -0.3  # DynamicRouter bias_raw 初始值
    router_decay_len: float = 400.0  # 长度衰减常数：effective_beta = beta * exp(-L/decay_len)
    use_no_chunk: bool = False
    max_seq_len: int = 2048

@dataclass
class LossConfig:
    """Loss function configuration"""
    lambda_recon: float = 0
    lambda_mlm: float = 1.5
    lambda_mdl: float = 0.5
    mdl_cost_base: float = 0.05
    mdl_delta: float = 0.5
    mdl_warmup_ratio: float = 0
    ignore_index: int = RNA_TOKEN_IDS["PAD"]
    strict_nan_check: bool = True

@dataclass 
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 64
    num_epochs: int = 300
    lr: float = 8e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    tau_init: float = 1.0
    tau_final: float = 0.3
    device: str = "cuda"
    grad_accum_steps: int = 1
    max_steps: Optional[int] = None
    
    # Scheduler Params
    warmup_ratio: float = 0.1
    warmup_start_factor: float = 0.1
    cosine_eta_min: float = 0.0
    
    # Finetuning params
    head_lr: float = 1e-3
    backbone_lr: float = 1e-4
    warmup_epochs: int = 0
    backbone_lr_scale: float = 0.1

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Base directory
    base_dir: str = _REPO_ROOT

    # Data paths — pretraining
    fasta_path: str = os.path.join(_REPO_ROOT, "data", "pre_random", "train_pre.fasta")
    # pkl_path: str = os.path.join(_REPO_ROOT, "data", "pre_random", "train_shards")
    pkl_path: str = os.path.join(_REPO_ROOT, "data", "pre_random", "train_shards","merged.pkl")

    # Data paths — finetuning (each can be a file or shard directory)
    ft_train_fasta: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "train_ft.fasta")
    ft_train_bppm: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "train_ft_shards","merged.pkl")
    ft_train_dotbracket: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "train_ft_dotbracket.txt")
    ft_val_fasta: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "val_ft.fasta")
    ft_val_bppm: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "val_ft_shards","merged.pkl")
    ft_val_dotbracket: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "val_ft_dotbracket.txt")
    ft_test_fasta: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "test_ft.fasta")
    ft_test_bppm: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "test_ft_shards","merged.pkl")
    ft_test_dotbracket: str = os.path.join(_REPO_ROOT, "data", "ft", "rna_families", "test_ft_dotbracket.txt")

    # Length buckets for per-bucket accuracy reporting (comma-separated sorted boundaries)
    length_buckets: str = "100,150,250,400"

    # Experiment name
    exp_name: str = 'test'
    
    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            "masking": self.masking.__dict__,
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "training": self.training.__dict__,
            "fasta_path": self.fasta_path,
            "pkl_path": self.pkl_path,
            "ft_train_fasta": self.ft_train_fasta,
            "ft_train_bppm": self.ft_train_bppm,
            "ft_train_dotbracket": self.ft_train_dotbracket,
            "ft_val_fasta": self.ft_val_fasta,
            "ft_val_bppm": self.ft_val_bppm,
            "ft_val_dotbracket": self.ft_val_dotbracket,
            "ft_test_fasta": self.ft_test_fasta,
            "ft_test_bppm": self.ft_test_bppm,
            "ft_test_dotbracket": self.ft_test_dotbracket,
            "length_buckets": self.length_buckets,
            "base_dir": self.base_dir,
            "exp_name": self.exp_name
        }

    @staticmethod
    def _filter_dataclass_kwargs(dataclass_type, raw_values: Dict[str, Any]) -> Dict[str, Any]:
        valid_keys = {item.name for item in fields(dataclass_type)}
        return {key: value for key, value in raw_values.items() if key in valid_keys}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Load from dictionary"""
        config = cls()
        if "masking" in config_dict:
            config.masking = MaskingConfig(**cls._filter_dataclass_kwargs(MaskingConfig, config_dict["masking"]))
        if "model" in config_dict:
            config.model = ModelConfig(**cls._filter_dataclass_kwargs(ModelConfig, config_dict["model"]))
        if "loss" in config_dict:
            config.loss = LossConfig(**cls._filter_dataclass_kwargs(LossConfig, config_dict["loss"]))
        if "training" in config_dict:
            config.training = TrainingConfig(**cls._filter_dataclass_kwargs(TrainingConfig, config_dict["training"]))
            
        config.fasta_path = config_dict.get("fasta_path", config.fasta_path)
        config.pkl_path = config_dict.get("pkl_path", config.pkl_path)
        for ft_key in ("ft_train_fasta", "ft_train_bppm", "ft_train_dotbracket",
                       "ft_val_fasta", "ft_val_bppm", "ft_val_dotbracket",
                       "ft_test_fasta", "ft_test_bppm", "ft_test_dotbracket"):
            setattr(config, ft_key, config_dict.get(ft_key, getattr(config, ft_key)))
        config.length_buckets = config_dict.get("length_buckets", config.length_buckets)
        config.base_dir = config_dict.get("base_dir", config.base_dir)
        config.exp_name = config_dict.get("exp_name", config.exp_name)
        return config

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Preset configurations

# Preset 1: Baseline (No MLM)
BASELINE_CONFIG = ExperimentConfig(
    masking=MaskingConfig(use_masking=False)
)

# Preset 2: Simple MLM
SIMPLE_MLM_CONFIG = ExperimentConfig(
    masking=MaskingConfig(
        use_masking=True,
        masking_type='simple',
        mask_prob=0.15
    )
)

# Preset 3: Pairing-Aware MLM
PAIRING_AWARE_MLM_CONFIG = ExperimentConfig(
    masking=MaskingConfig(
        use_masking=True,
        masking_type='pairing_aware',
        mask_prob=0.15,
        coupled_prob=0.5,
        pairing_threshold=0.3
    )
)

def resolve_exp_name_with_mode_suffix(
    base_name: str,
    *,
    no_chunk: bool = False,
    fixed_router: bool = False,
    chunk_size: Optional[int] = None,
) -> str:
    """Append a chunking-mode suffix to ``base_name`` so that ablation modes
    (no_chunk / fixed_router / dynamic) cannot silently overwrite each other's
    experiment directory.

    Returns ``base_name`` unchanged if it already ends with the suffix that
    would normally be applied (so the function is idempotent and safe to call
    repeatedly). The default ``dynamic`` mode does not append anything.
    """
    if no_chunk:
        suffix = "noChunk"
    elif fixed_router:
        suffix = f"fixed_cs{chunk_size}" if chunk_size else "fixed"
    else:
        return base_name
    if base_name.endswith(f"_{suffix}"):
        return base_name
    return f"{base_name}_{suffix}"

def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate configuration legality, return error list"""
    errors = []
    
    # Validate masking type
    valid_masking_types = ['simple', 'pairing_aware']
    if config.masking.masking_type not in valid_masking_types:
        errors.append(f"Invalid masking_type: {config.masking.masking_type}. Must be one of {valid_masking_types}")
        
    # Validate probabilities
    probs = [
        config.masking.mask_prob, 
        config.masking.coupled_prob,
        config.masking.pairing_threshold,
        config.masking.replace_prob,
        config.masking.random_prob
    ]
    for p in probs:
        if not (0.0 <= p <= 1.0):
            errors.append(f"Probability value {p} out of range [0, 1]")
            
    # Validate loss weights
    loss_weights = {
        "lambda_recon": config.loss.lambda_recon,
        "lambda_mlm": config.loss.lambda_mlm,
        "lambda_mdl": config.loss.lambda_mdl,
    }
    for name, val in loss_weights.items():
        if val < 0:
            errors.append(f"{name} must be non-negative, got {val}")
    if all(v == 0 for v in loss_weights.values()):
        errors.append(
            "All loss weights are zero (lambda_recon, lambda_mlm, lambda_mdl). "
            "At least one must be non-zero for the model to receive training signal."
        )

    # Validate paths (warn only)
    if not os.path.exists(config.fasta_path):
        pass # print(f"Warning: Fasta path not found: {config.fasta_path}")
    if not os.path.exists(config.pkl_path):
        pass # print(f"Warning: PKL path not found: {config.pkl_path}")
        
    return errors
