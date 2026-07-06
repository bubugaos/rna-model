import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCosineRouter(nn.Module):
    """Cosine dissimilarity router with learned Q/K projections."""

    def __init__(self, dim):
        super(AttentionCosineRouter, self).__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(self, h_local, mask=None):
        """
        Args:
            h_local (torch.Tensor): 局部编码特征，形状为 (Batch, L, dim)。
            mask (torch.Tensor, optional): 序列掩码，形状为 (Batch, L)。
            
        Returns:
            c_scores (torch.Tensor): 计算出的多尺度相似度得分，形状为 (Batch, L-1)。
        """
        batch_size, seq_len, _ = h_local.shape
        if seq_len <= 1:
            return h_local.new_zeros(batch_size, 0)

        q = self.q_proj(h_local[:, 1:, :])
        k = self.k_proj(h_local[:, :-1, :])
        c_scores = 0.5 * (1.0 - F.cosine_similarity(q, k, dim=-1))
        if mask is not None:
            valid_boundary_mask = mask[:, :-1] * mask[:, 1:]
            c_scores = c_scores * valid_boundary_mask.to(dtype=c_scores.dtype)
        return c_scores

class FixedChunkRouter(nn.Module):
    """Fixed-size chunk baseline."""

    def __init__(self, chunk_size=8):
        super(FixedChunkRouter, self).__init__()
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            self.chunk_size = 1

    def forward(self, h_local, cross_pair_sum=None, mask=None):
        batch_size, seq_len, _ = h_local.size()
        device = h_local.device
        dtype = h_local.dtype

        boundary_mask = torch.zeros((batch_size, max(seq_len - 1, 0)), device=device, dtype=dtype)

        if seq_len >= 2:
            cut_pos = torch.arange(self.chunk_size - 1, seq_len - 1, self.chunk_size, device=device)
            if cut_pos.numel() > 0:
                boundary_mask[:, cut_pos] = 1.0

        if mask is None:
            valid_boundary_mask = torch.ones_like(boundary_mask, dtype=dtype, device=device)
        else:
            token_mask = mask.to(device=device, dtype=dtype)
            valid_boundary_mask = token_mask[:, :-1] * token_mask[:, 1:]
            boundary_mask = boundary_mask * valid_boundary_mask

        boundary_probs = boundary_mask
        boundary_logits = boundary_mask

        segment_ids = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
        segment_ids[:, 1:] = torch.cumsum(boundary_mask, dim=1).to(dtype=torch.long)
        if mask is not None:
            segment_ids = segment_ids.masked_fill(mask == 0, -1)

        expected_segments = 1.0 + (boundary_probs * valid_boundary_mask).sum(dim=1)

        b_scores = torch.zeros_like(boundary_probs)
        c_scores = torch.zeros_like(boundary_probs)

        return {
            "boundary_mask": boundary_mask,
            "boundary_logits": boundary_logits,
            "boundary_probs": boundary_probs,
            "segment_ids": segment_ids,
            "expected_segments": expected_segments,
            "b_scores": b_scores,
            "c_scores": c_scores
        }

class DynamicRouter(nn.Module):
    """Fuse semantic dissimilarity and BPPM break scores into boundary probabilities."""

    @staticmethod
    def _inverse_softplus(value):
        value = float(value)
        if value <= 0.0:
            return torch.tensor(-20.0)
        return torch.log(torch.expm1(torch.tensor(value)))

    def __init__(self, dim, beta=1.0, bias_init=-0.3, heads=4, decay_len=400.0):
        super(DynamicRouter, self).__init__()
        self.similarity_router = AttentionCosineRouter(dim)
        self.alpha_raw = nn.Parameter(torch.tensor(1.0))
        self.use_bppm = float(beta) > 0.0
        self.beta_raw = nn.Parameter(self._inverse_softplus(beta))
        self.logit_scale_raw = nn.Parameter(torch.tensor(0.1))
        self.bias_raw = nn.Parameter(torch.tensor(float(bias_init)))
        self.decay_len = float(decay_len)
        self.tau = 1.0

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, h_local, cross_pair_sum=None, mask=None):
        batch_size, seq_len, _ = h_local.size()
        if seq_len <= 1:
            empty = h_local.new_zeros((batch_size, 0))
            segment_ids = torch.zeros((batch_size, seq_len), device=h_local.device, dtype=torch.long)
            if mask is not None:
                segment_ids = segment_ids.masked_fill(mask == 0, -1)
            return {
                "boundary_mask": empty,
                "boundary_logits": empty,
                "boundary_probs": empty,
                "segment_ids": segment_ids,
                "expected_segments": torch.ones(batch_size, device=h_local.device, dtype=h_local.dtype),
                "b_scores": empty,
                "c_scores": empty
            }
        
        c_scores = self.similarity_router(h_local, mask)

        # Use precomputed cross-pair sum vector instead of full BPPM matrix
        if cross_pair_sum is not None:
            u_scores = cross_pair_sum.clamp(min=0.0)
            if mask is not None:
                valid_boundary_mask = mask[:, :-1] * mask[:, 1:]
                u_scores = u_scores * valid_boundary_mask.to(dtype=u_scores.dtype)
        else:
            u_scores = h_local.new_zeros((batch_size, seq_len - 1))
        eps = 1e-8
        if mask is None:
            lengths = torch.full((batch_size,), seq_len, device=h_local.device, dtype=torch.long)
        else:
            lengths = mask.sum(dim=1).to(dtype=torch.long).clamp(min=1, max=seq_len)
        cut_sizes = torch.arange(1, seq_len, device=h_local.device, dtype=u_scores.dtype).unsqueeze(0)
        lengths_float = lengths.unsqueeze(1).to(dtype=u_scores.dtype)
        cross_pair_count = cut_sizes * (lengths_float - cut_sizes)
        b_scores = (u_scores / cross_pair_count.clamp(min=1.0)).clamp(0.0, 1.0)

        alpha = F.softplus(self.alpha_raw) + 1e-6
        beta = F.softplus(self.beta_raw) if self.use_bppm else h_local.new_tensor(0.0)

        # Length-dependent structural weight: long RNA sequences have
        # less reliable BPPM (partition function numerical drift, diffuse
        # pairing probabilities).  The exponential decay smoothly shifts
        # boundary decisions from structural (b_scores) to semantic
        # (c_scores) as sequence length increases.
        if mask is not None:
            eff_lengths = mask.sum(dim=1).to(dtype=beta.dtype).clamp(min=1)
        else:
            eff_lengths = torch.full(
                (batch_size,), seq_len, device=h_local.device, dtype=beta.dtype
            )
        length_weight = torch.exp(-eff_lengths / self.decay_len)  # (B,)
        effective_beta = beta * length_weight                     # (B,)

        d_scores = alpha * c_scores - effective_beta.unsqueeze(1) * b_scores + self.bias_raw
        
        valid_boundary_mask = None
        if mask is not None:
            valid_boundary_mask = mask[:, :-1] * mask[:, 1:]
            d_scores = d_scores.masked_fill(valid_boundary_mask == 0, -1e9)
            b_scores = b_scores * valid_boundary_mask.to(dtype=b_scores.dtype)
            c_scores = c_scores * valid_boundary_mask.to(dtype=c_scores.dtype)

        logit_scale = F.softplus(self.logit_scale_raw) + 1e-6
        effective_tau = max(float(self.tau), 1e-4)
        boundary_probs = torch.sigmoid(d_scores / (logit_scale * effective_tau))

        # ── Vanilla Straight-Through Estimator (STE) ──
        # Forward: hard {0,1} decision matching old behavior.
        # Backward: gradient flows through boundary_probs as if it were a
        # continuous relaxation, so downstream losses that consume
        # boundary_mask (e.g. STE-friendly downsamplers or future regularizers)
        # can update the router parameters (alpha_raw / beta_raw / logit_scale /
        # similarity_router weights).
        #
        # Limitation: segment_ids = cumsum(boundary_mask).long() and the gather
        # used in Downsampler are still non-differentiable. So in the current
        # pipeline the dominant learning signal for router parameters still flows
        # through (1) compression_loss via boundary_probs and (2) the EMA path
        # in Dechunker via transition_probs <- boundary_probs (both already
        # differentiable). STE here is a prerequisite for future soft-downsample
        # / Surprise / MDL extensions; it does not by itself unlock a new
        # learning path through the hard segment assignment.
        boundary_hard = (boundary_probs >= 0.5).to(dtype=boundary_probs.dtype)
        boundary_mask = boundary_hard + (boundary_probs - boundary_probs.detach())

        if valid_boundary_mask is not None:
            valid_boundary_mask_float = valid_boundary_mask.to(dtype=boundary_probs.dtype)
            boundary_probs = boundary_probs * valid_boundary_mask_float
            boundary_mask = boundary_mask * valid_boundary_mask_float
        else:
            valid_boundary_mask_float = torch.ones((batch_size, seq_len - 1), device=h_local.device, dtype=boundary_probs.dtype)
            
        segment_ids = torch.zeros((batch_size, seq_len), device=boundary_mask.device, dtype=torch.long)
        segment_ids[:, 1:] = torch.cumsum(boundary_mask, dim=1).long()
        
        if mask is not None:
            segment_ids = segment_ids.masked_fill(mask == 0, -1)

        expected_segments = 1.0 + (boundary_probs * valid_boundary_mask_float).sum(dim=1)

        return {
            "boundary_mask": boundary_mask,
            "boundary_logits": d_scores,
            "boundary_probs": boundary_probs,
            "segment_ids": segment_ids,
            "expected_segments": expected_segments,
            "b_scores": b_scores,
            "c_scores": c_scores
        }

MultiScaleSimilarityRouter = AttentionCosineRouter


