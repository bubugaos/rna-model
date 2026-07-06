import torch
from typing import Tuple, Optional
import numpy as np
from src.config import RNA_TOKEN_IDS, RNA_INPUT_VOCAB_SIZE, RNA_BASE_VOCAB_SIZE

def _build_mask_probability_matrix(x: torch.Tensor, mask_prob: float, pad_token_id: Optional[int]) -> torch.Tensor:
    device = x.device
    batch_size, seq_len = x.shape
    if pad_token_id is None:
        valid_lengths = torch.full((batch_size, 1), float(seq_len), device=device)
    else:
        valid_lengths = (x != pad_token_id).sum(dim=1, keepdim=True).float()
    dynamic_mask_prob = torch.where(
        valid_lengths < 30.0,
        torch.full_like(valid_lengths, 0.05),
        torch.where(
            valid_lengths < 60.0,
            torch.full_like(valid_lengths, 0.10),
            torch.full_like(valid_lengths, float(mask_prob)),
        ),
    )
    prob_matrix = dynamic_mask_prob.expand(-1, seq_len).clone()
    if pad_token_id is not None:
        prob_matrix[x == pad_token_id] = 0.0
    return prob_matrix

def _apply_token_replacement(
    x: torch.Tensor,
    mask_positions: torch.Tensor,
    mask_token_id: int,
    replace_prob: float,
    random_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    masked_x = x.clone()
    labels = x.clone()
    if not mask_positions.any():
        return masked_x, mask_positions, labels
    indices_replaced = torch.bernoulli(torch.full(x.shape, replace_prob, device=device)).bool() & mask_positions
    masked_x[indices_replaced] = mask_token_id
    indices_random = (
        torch.bernoulli(torch.full(x.shape, random_prob / max(1.0 - replace_prob, 1e-6), device=device)).bool()
        & mask_positions
        & ~indices_replaced
    )
    random_tokens = torch.randint(0, RNA_BASE_VOCAB_SIZE, x.shape, device=device)
    masked_x[indices_random] = random_tokens[indices_random]
    return masked_x, mask_positions, labels

class BaseMasking:
    def __call__(self, x: torch.Tensor, top1_partner: Optional[torch.Tensor] = None,
                 top1_prob: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, L) sequence
            top1_partner: (Batch, L) index of max-prob pairing partner per position
            top1_prob: (Batch, L) probability of that partner
        Returns:
            masked_x: (Batch, L) Masked sequence
            mask_positions: (Batch, L) bool tensor, True where masked
            labels: (Batch, L) Original tokens for loss calculation
        """
        raise NotImplementedError

class SimpleMasking(BaseMasking):
    """
    Standard BERT-style masking: 15% random masking.
    80% -> [MASK], 10% -> random, 10% -> original
    """
    def __init__(self, mask_prob=0.15, mask_token_id=RNA_TOKEN_IDS["MASK"], vocab_size=RNA_INPUT_VOCAB_SIZE, 
                 replace_prob=0.8, random_prob=0.1, pad_token_id=RNA_TOKEN_IDS["PAD"]):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.pad_token_id = pad_token_id
        
    def __call__(self, x: torch.Tensor, top1_partner: Optional[torch.Tensor] = None,
                 top1_prob: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob_matrix = _build_mask_probability_matrix(x, self.mask_prob, self.pad_token_id)
        mask_positions = torch.bernoulli(prob_matrix).bool()
        return _apply_token_replacement(
            x,
            mask_positions,
            self.mask_token_id,
            self.replace_prob,
            self.random_prob,
        )

class PairingAwareMasking(BaseMasking):
    def __init__(
        self,
        mask_prob=0.15,
        coupled_prob=0.5,
        pairing_threshold=0.3,
        mask_token_id=RNA_TOKEN_IDS["MASK"],
        vocab_size=RNA_INPUT_VOCAB_SIZE,
        replace_prob=0.8,
        random_prob=0.1,
        pad_token_id=RNA_TOKEN_IDS["PAD"],
    ):
        self.mask_prob = mask_prob
        self.coupled_prob = coupled_prob
        self.pairing_threshold = pairing_threshold
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.pad_token_id = pad_token_id

    def __call__(self, x: torch.Tensor, top1_partner: Optional[torch.Tensor] = None,
                 top1_prob: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if top1_partner is None or top1_prob is None:
            raise ValueError("PairingAwareMasking requires top1_partner and top1_prob vectors")
        if top1_partner.ndim != 2 or top1_prob.ndim != 2:
            raise ValueError(f"Expected 2D vectors (batch, seq_len), got ndim={top1_partner.ndim}")
        if top1_partner.shape[0] != x.shape[0] or top1_partner.shape[1] != x.shape[1]:
            raise ValueError(f"Vector shape {tuple(top1_partner.shape)} does not match input {tuple(x.shape)}")

        device = x.device
        prob_matrix = _build_mask_probability_matrix(x, self.mask_prob, self.pad_token_id)
        selected = torch.bernoulli(prob_matrix).bool()

        pair_partner = top1_partner.to(device=device, dtype=torch.long)
        max_pair_prob = top1_prob.to(device=device, dtype=torch.float32)

        if self.pad_token_id is None:
            valid_tokens = torch.ones_like(x, dtype=torch.bool)
        else:
            valid_tokens = x != self.pad_token_id

        couple_rand = torch.rand_like(prob_matrix)
        couple_mask = (
            selected
            & valid_tokens
            & (max_pair_prob >= self.pairing_threshold)
            & (couple_rand < self.coupled_prob)
        )
        partner_mask = torch.zeros_like(selected, dtype=torch.int32)
        partner_mask.scatter_add_(1, pair_partner.clamp_min(0), couple_mask.int())
        mask_positions = (selected | (partner_mask > 0)) & valid_tokens

        return _apply_token_replacement(
            x,
            mask_positions,
            self.mask_token_id,
            self.replace_prob,
            self.random_prob,
        )

if __name__ == "__main__":
    print("Testing Masking Strategies...")
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randint(0, 4, (3, 80))
    x[0, 25:] = 4
    x[1, 50:] = 4
    bppm = torch.rand(3, 80, 80)

    simple_masker = SimpleMasking(mask_prob=0.15)
    rates = torch.zeros(3, dtype=torch.float)
    repeats = 200
    for _ in range(repeats):
        masked_x, mask_pos, labels = simple_masker(x)
        for i in range(3):
            valid = x[i] != simple_masker.pad_token_id
            rates[i] += mask_pos[i][valid].float().mean()
    rates /= repeats
    print(f"Simple Masking Rates (len<30, <60, >=60): {[round(r.item(), 3) for r in rates]}")
    assert abs(rates[0].item() - 0.05) < 0.02
    assert abs(rates[1].item() - 0.10) < 0.02
    assert abs(rates[2].item() - 0.15) < 0.02
    assert not mask_pos[0, 25:].any(), "Padding should not be masked"

    bppm.zero_()
    bppm[:, 10, 60] = 0.9
    bppm[:, 60, 10] = 0.9
    # Derive top1_partner and top1_prob from bppm
    bppm_no_diag = bppm.clone()
    bppm_no_diag.diagonal(dim1=1, dim2=2).fill_(0)
    top1_prob, top1_partner = bppm_no_diag.max(dim=-1)
    pairing_masker = PairingAwareMasking(mask_prob=0.2, coupled_prob=1.0, pairing_threshold=0.3)
    coupled_counts = torch.zeros(3, dtype=torch.float)
    repeats = 200
    for _ in range(repeats):
        _, mask_pos, _ = pairing_masker(x, top1_partner, top1_prob)
        coupled_counts += mask_pos[:, 10].float() * mask_pos[:, 60].float()
    coupled_counts /= repeats
    print(f"Pairing-Aware Coupled Rate: {[round(v.item(), 3) for v in coupled_counts]}")
    assert coupled_counts[2].item() > 0.15
    assert not mask_pos[0, 25:].any(), "Padding should not be masked"

    print("All tests passed!")
