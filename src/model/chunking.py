import torch
import torch.nn as nn


class Downsampler(nn.Module):
    """Select the first token of each chunk as its chunk representation."""

    def forward(self, h_local, segment_ids, mask=None):
        if segment_ids is None:
            raise ValueError("segment_ids is required for Downsampler")

        batch_size, seq_len, dim = h_local.shape
        valid_tokens = segment_ids >= 0
        if mask is not None:
            valid_tokens = valid_tokens & mask.bool()

        if not valid_tokens.any():
            chunk_repr = h_local.new_zeros(batch_size, 1, dim)
            chunk_mask = torch.zeros(batch_size, 1, device=h_local.device, dtype=torch.bool)
            chunk_starts = torch.zeros(batch_size, 1, device=h_local.device, dtype=torch.long)
            return {
                "chunk_repr": chunk_repr,
                "chunk_mask": chunk_mask,
                "src_key_padding_mask": ~chunk_mask,
                "chunk_starts": chunk_starts,
                "chunk_counts": torch.zeros(batch_size, device=h_local.device, dtype=torch.long),
            }

        safe_ids = segment_ids.masked_fill(~valid_tokens, 0)
        chunk_counts = torch.where(
            valid_tokens.any(dim=1),
            safe_ids.max(dim=1).values + 1,
            torch.zeros(batch_size, device=h_local.device, dtype=torch.long),
        )
        max_chunks = max(int(chunk_counts.max().item()), 1)

        token_positions = torch.arange(seq_len, device=h_local.device).unsqueeze(0).expand(batch_size, -1)
        large_value = seq_len

        # Vectorized: scatter token positions to chunk slots and take the min.
        # Invalid tokens get large_value so they don't affect the minimum.
        scatter_positions = torch.where(valid_tokens, token_positions, torch.full_like(token_positions, large_value))
        chunk_starts = torch.full(
            (batch_size, max_chunks),
            large_value,
            device=h_local.device,
            dtype=torch.long,
        )
        chunk_starts.scatter_reduce_(1, safe_ids, scatter_positions, reduce="amin", include_self=True)

        chunk_mask = token_positions[:, :max_chunks] < chunk_counts.unsqueeze(1)
        safe_chunk_starts = chunk_starts.masked_fill(~chunk_mask, 0)
        gather_index = safe_chunk_starts.unsqueeze(-1).expand(-1, -1, dim)
        chunk_repr = torch.gather(h_local, 1, gather_index)
        chunk_repr = chunk_repr.masked_fill(~chunk_mask.unsqueeze(-1), 0.0)

        return {
            "chunk_repr": chunk_repr,
            "chunk_mask": chunk_mask,
            "src_key_padding_mask": ~chunk_mask,
            "chunk_starts": safe_chunk_starts,
            "chunk_counts": chunk_counts,
        }


class Dechunker(nn.Module):
    """EMA-smooth chunk states, upsample by segment ids, then fuse with non-identity local context."""

    def __init__(self, dim, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        self.struct_proj = nn.Linear(1, dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _compress_boundary_probs(boundary_probs, segment_ids, max_chunks):
        batch_size = boundary_probs.size(0)
        if max_chunks <= 1 or boundary_probs.size(1) == 0:
            return boundary_probs.new_zeros((batch_size, max(max_chunks - 1, 0)))

        prev_ids = segment_ids[:, :-1]
        next_ids = segment_ids[:, 1:]
        is_transition = (prev_ids >= 0) & (next_ids >= 0) & (next_ids > prev_ids)

        transition_probs = boundary_probs.new_zeros((batch_size, max_chunks - 1))
        if not is_transition.any():
            return transition_probs

        batch_index = torch.arange(batch_size, device=segment_ids.device).unsqueeze(1).expand_as(prev_ids)
        transition_index = (next_ids[is_transition] - 1).long()
        transition_probs[batch_index[is_transition], transition_index] = boundary_probs[is_transition]
        return transition_probs

    def _apply_ema_smoothing(self, chunk_repr, transition_probs, src_key_padding_mask):
        batch_size, max_chunks, _ = chunk_repr.shape
        if max_chunks <= 1:
            return chunk_repr.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        smoothed = []
        prev_state = chunk_repr[:, 0, :]
        smoothed.append(prev_state)

        for idx in range(1, max_chunks):
            confidence = transition_probs[:, idx - 1].unsqueeze(-1)
            current_state = confidence * chunk_repr[:, idx, :] + (1.0 - confidence) * prev_state
            valid = (~src_key_padding_mask[:, idx]).unsqueeze(-1)
            current_state = torch.where(valid, current_state, torch.zeros_like(current_state))
            smoothed.append(current_state)
            prev_state = current_state

        smoothed = torch.stack(smoothed, dim=1)
        return smoothed.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

    def forward(self, chunk_repr, h_local, segment_ids, boundary_probs, src_key_padding_mask, mask=None, bppm_row=None):
        if segment_ids is None:
            raise ValueError("segment_ids is required for Dechunker")

        max_chunks = chunk_repr.size(1)
        transition_probs = self._compress_boundary_probs(boundary_probs, segment_ids, max_chunks)
        smoothed_chunks = self._apply_ema_smoothing(chunk_repr, transition_probs, src_key_padding_mask)

        valid_tokens = segment_ids >= 0
        if mask is not None:
            valid_tokens = valid_tokens & mask.bool()

        safe_ids = segment_ids.masked_fill(~valid_tokens, 0).clamp(min=0, max=max(max_chunks - 1, 0))
        gather_index = safe_ids.unsqueeze(-1).expand(-1, -1, chunk_repr.size(-1))
        upsampled = torch.gather(smoothed_chunks, 1, gather_index)
        upsampled = upsampled.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)

        batch_size, seq_len, dim = h_local.shape
        device = h_local.device
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        local_ctx = self.pos_embed(pos_ids)
        if bppm_row is not None:
            local_ctx = local_ctx + self.struct_proj(bppm_row.unsqueeze(-1).to(local_ctx.dtype))

        fused = self.fuse(torch.cat([upsampled, local_ctx], dim=-1))
        fused = fused.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)

        return {
            "transition_probs": transition_probs,
            "smoothed_chunks": smoothed_chunks,
            "upsampled": upsampled,
            "token_features": fused,
        }
