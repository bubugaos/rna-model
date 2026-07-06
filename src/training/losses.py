import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import RNA_OUTPUT_VOCAB_SIZE

class RNALMCriterion(nn.Module):
    """Loss contract for the refactored chunking pipeline."""

    def __init__(
        self,
        vocab_size=RNA_OUTPUT_VOCAB_SIZE,
        lambda_recon=1.0,
        lambda_mlm=1.0,
        lambda_mdl=0.0,
        mdl_cost_base=0.05,
        mdl_delta=0.5,
        ignore_index=4,
        strict_nan_check=True,
    ):
        super(RNALMCriterion, self).__init__()
        self.vocab_size = vocab_size
        self.lambda_recon = lambda_recon
        self.lambda_mlm = lambda_mlm
        self.lambda_mdl = lambda_mdl
        self.mdl_cost_base = mdl_cost_base
        self.mdl_delta = mdl_delta
        self.mdl_weight = 0.0
        self.ignore_index = ignore_index
        self.strict_nan_check = bool(strict_nan_check)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _compute_mlm_loss(self, recon_logits, labels, mask_positions, mask=None):
        if mask_positions is None or not mask_positions.any():
            return torch.tensor(0.0, device=recon_logits.device, requires_grad=True)
        # Defensive: ensure mask_positions only includes valid (non-padding) positions
        if mask is not None:
            mask_positions = mask_positions & mask.bool()
            if not mask_positions.any():
                return torch.tensor(0.0, device=recon_logits.device, requires_grad=True)
        masked_logits = recon_logits[mask_positions]
        masked_labels = labels[mask_positions]
        if masked_logits.numel() == 0:
            return torch.tensor(0.0, device=recon_logits.device, requires_grad=True)
        mlm_loss = F.cross_entropy(
            masked_logits,
            masked_labels,
            ignore_index=self.ignore_index
        )
        return mlm_loss

    def _compute_recon_loss(self, recon_logits, x, mask, mask_positions=None):
        if mask is not None:
            active_positions = mask.bool()
        else:
            active_positions = torch.ones_like(x, dtype=torch.bool)

        if mask_positions is not None:
            active_positions = active_positions & (~mask_positions)

        if not active_positions.any():
            return torch.tensor(0.0, device=recon_logits.device, requires_grad=True)

        if recon_logits[active_positions].numel() == 0:
            return torch.tensor(0.0, device=recon_logits.device, requires_grad=True)

        return self.ce_loss(recon_logits[active_positions], x[active_positions])

    def _compute_mdl_loss(self, boundary_probs, smoothed_chunks, segment_ids, mask, bppm_row_sum):
        """MDL boundary loss: penalize boundaries whose removal wouldn't hurt reconstruction.

        Proxy for information gain: cosine distance between adjacent smoothed chunk
        representations.  If two adjacent chunks are nearly identical after the
        LatentTransformer, the boundary between them is redundant (Occam's Razor).

        Structure-aware cost: boundaries inside stems (high BPPM on both sides) need
        stronger justification, reflecting the RNA biological prior that base-paired
        regions form functional units that shouldn't be split.
        """
        if boundary_probs is None or boundary_probs.numel() == 0:
            return torch.tensor(0.0, device=mask.device if mask is not None else "cpu", requires_grad=True)

        if smoothed_chunks is None:
            return torch.tensor(0.0, device=boundary_probs.device, requires_grad=True)

        B, C, _ = smoothed_chunks.shape
        if C <= 1:
            return torch.tensor(0.0, device=boundary_probs.device, requires_grad=True)

        # ── 1. Per-chunk-boundary information gain ──
        # Cosine distance between adjacent smoothed chunks.
        # High distance → chunks encode different information → boundary is justified.
        # Low distance  → chunks are similar → boundary is redundant.
        chunk_i = F.normalize(smoothed_chunks[:, :-1, :], dim=-1)   # (B, C-1, D)
        chunk_j = F.normalize(smoothed_chunks[:, 1:, :], dim=-1)    # (B, C-1, D)
        cosine_sim = (chunk_i * chunk_j).sum(dim=-1)                 # (B, C-1)
        chunk_gain = 0.5 * (1.0 - cosine_sim)                        # (B, C-1), range [0, 1]

        # ── 2. Map chunk-level gain to token-level boundary positions ──
        prev_ids = segment_ids[:, :-1]   # (B, L-1)
        next_ids = segment_ids[:, 1:]    # (B, L-1)
        is_boundary = (next_ids >= 0) & (prev_ids >= 0) & (next_ids > prev_ids)

        if mask is not None:
            token_valid = (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
        else:
            token_valid = torch.ones_like(is_boundary, dtype=torch.bool)

        eval_mask = is_boundary & token_valid
        if not eval_mask.any():
            return torch.tensor(0.0, device=boundary_probs.device, requires_grad=True)

        chunk_boundary_idx = (next_ids - 1).clamp(0, C - 2)  # (B, L-1)
        token_gain = torch.gather(chunk_gain, 1, chunk_boundary_idx)  # (B, L-1)

        # ── 3. Structure-aware boundary cost ──
        if bppm_row_sum is not None and self.mdl_delta > 0:
            bppm_row = bppm_row_sum.to(dtype=smoothed_chunks.dtype)
            # Mask padding before normalization so all-zero rows stay zero
            if mask is not None:
                bppm_row = bppm_row * mask.to(dtype=bppm_row.dtype)
            row_max = bppm_row.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
            bppm_row = bppm_row / row_max
            stem_score = bppm_row[:, :-1] * bppm_row[:, 1:]                      # (B, L-1)
            cost = self.mdl_cost_base * (1.0 + self.mdl_delta * stem_score)     # (B, L-1)
        else:
            cost = torch.full_like(token_gain, float(self.mdl_cost_base))

        # ── 4. MDL loss: penalize boundary_prob where gain < cost ──
        # boundary_probs * relu(cost - gain): only active for existing boundaries
        # where the chunk representations don't justify the split.
        boundary_term = (boundary_probs * eval_mask.to(dtype=boundary_probs.dtype)
                         * F.relu(cost - token_gain))
        n_valid = eval_mask.float().sum().clamp(min=1.0)
        mdl_loss = boundary_term.sum() / n_valid

        if self.strict_nan_check and (torch.isnan(mdl_loss) or torch.isinf(mdl_loss)):
            raise RuntimeError("MDL loss is NaN or Inf.")
        return mdl_loss

    def forward(self, model_output, x, bppm_row_sum, mask, mask_positions=None, mlm_labels=None):
        recon_loss = self._compute_recon_loss(model_output["recon_logits"], x, mask, mask_positions=mask_positions)

        if mask_positions is not None and mlm_labels is not None:
            mlm_loss = self._compute_mlm_loss(
                model_output["recon_logits"],
                mlm_labels,
                mask_positions,
                mask=mask
            )
        else:
            mlm_loss = torch.tensor(0.0, device=x.device)

        mdl_loss = self._compute_mdl_loss(
            model_output.get("boundary_probs"),
            model_output.get("smoothed_chunks"),
            model_output.get("segment_ids"),
            mask,
            bppm_row_sum,
        )
        total_loss = (
            self.lambda_recon * recon_loss
            + self.lambda_mlm * mlm_loss
            + self.lambda_mdl * self.mdl_weight * mdl_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "mlm_loss": mlm_loss,
            "mdl_loss": mdl_loss,
        }

    def set_mdl_weight(self, weight):
        self.mdl_weight = float(weight)

if __name__ == "__main__":
    print("Testing RNALMCriterion...")
    criterion = RNALMCriterion(lambda_mlm=1.0, lambda_mdl=0.5)

    batch_size = 2
    seq_len = 16
    vocab_size = 6

    x = torch.randint(0, 4, (batch_size, seq_len))
    bppm = torch.rand(batch_size, seq_len, seq_len)
    bppm = (bppm + bppm.transpose(1, 2)) / 2
    bppm_row_sum = bppm.sum(dim=-1)
    mask = torch.ones(batch_size, seq_len)

    segment_ids = torch.arange(seq_len // 4).repeat_interleave(4).unsqueeze(0).repeat(batch_size, 1)

    model_output = {
        "boundary_mask": torch.rand(batch_size, seq_len - 1),
        "boundary_probs": torch.rand(batch_size, seq_len - 1),
        "recon_logits": torch.randn(batch_size, seq_len, vocab_size),
        "segment_ids": segment_ids,
        "smoothed_chunks": torch.randn(batch_size, 4, 64),
    }

    # 1. Test without MLM
    print("\n1. Without MLM:")
    losses = criterion(model_output, x, bppm_row_sum, mask)
    print(f"MLM Loss: {losses['mlm_loss'].item()}")
    assert losses['mlm_loss'].item() == 0.0

    # 2. Test with MLM
    print("\n2. With MLM:")
    mask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_positions[:, :2] = True  # Mask first 2
    mlm_labels = x.clone()

    losses = criterion(model_output, x, bppm_row_sum, mask, mask_positions, mlm_labels)
    print(f"MLM Loss: {losses['mlm_loss'].item()}")
    assert losses['mlm_loss'].item() > 0.0
    print(f"MDL Loss: {losses['mdl_loss'].item()}")
    assert losses['mdl_loss'].item() >= 0.0

    print("\nLoss Tests Passed!")
