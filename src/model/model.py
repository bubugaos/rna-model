import torch
import torch.nn as nn
from .local_encoder import LocalEncoder
from .dynamic_router import DynamicRouter, FixedChunkRouter
from .chunking import Downsampler, Dechunker
from .latent_transformer import LatentTransformer
from .decoder import ReconDecoder
from src.config import RNA_INPUT_VOCAB_SIZE, RNA_OUTPUT_VOCAB_SIZE


class RNADynamicModel(nn.Module):
    """RNA dynamic chunking model with a shared deep backbone across three modes.

    Architecture (Option B, shared backbone):
        x -> LocalEncoder(local_num_layers)
          -> BPPM struct injection
          -> [optional: Router + Downsampler]            (skipped if use_no_chunk)
          -> LatentTransformer(latent_num_layers)        (shared, runs on token-level
                                                          or chunk-level features)
          -> [optional: Dechunker upsample]              (skipped if use_no_chunk)
          -> ReconDecoder                                (always)
        Classifier always pools h_latent over valid positions (token positions for
        no_chunk, chunk positions for fixed/dynamic).

    This unifies the three ablation modes (no_chunk / fixed / dynamic) so that the
    only difference is the presence and design of the chunking module. All other
    components (LocalEncoder, BPPM injection, LatentTransformer, classifier and
    reconstruction head) are shared, which makes the comparison a controlled
    experiment over chunking strategies only.
    """

    def __init__(self,
                 input_vocab_size=RNA_INPUT_VOCAB_SIZE,
                 output_vocab_size=RNA_OUTPUT_VOCAB_SIZE,
                 embed_dim=256,
                 nhead=8,
                 num_layers=3,
                 local_num_layers=None,
                 latent_num_layers=None,
                 dim_feedforward=1024,
                 beta=1.0,
                 router_bias_init=-0.3,
                 router_decay_len=400.0,
                 num_classes=10,
                 use_fixed_router=False,
                 chunk_size=8,
                 use_no_chunk=False,
                 use_struct_injection=True,
                 dropout=0.1,
                 max_seq_len=2048):
        super(RNADynamicModel, self).__init__()
        self.use_no_chunk = bool(use_no_chunk)
        self.use_fixed_router = bool(use_fixed_router)
        self.use_struct_injection = bool(use_struct_injection)
        self.chunk_size = int(chunk_size)
        local_num_layers = int(num_layers if local_num_layers is None else local_num_layers)
        latent_num_layers = int(num_layers if latent_num_layers is None else latent_num_layers)

        self.local_encoder = LocalEncoder(
            vocab_size=input_vocab_size,
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=local_num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )

        # Chunking sub-modules are only instantiated for chunked modes. The
        # shared backbone (latent_transformer + recon_decoder + classifier) is
        # always present so all three modes see identical capacity downstream
        # of the chunking decision.
        if not self.use_no_chunk:
            if self.use_fixed_router:
                self.dynamic_router = FixedChunkRouter(chunk_size=self.chunk_size)
            else:
                self.dynamic_router = DynamicRouter(dim=embed_dim, beta=beta, bias_init=router_bias_init, decay_len=router_decay_len)

            self.downsampler = Downsampler()
            self.dechunker = Dechunker(dim=embed_dim, dropout=dropout, max_seq_len=max_seq_len)

        self.latent_transformer = LatentTransformer(
            dim=embed_dim,
            nhead=nhead,
            num_layers=latent_num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.recon_decoder = ReconDecoder(
            dim=embed_dim,
            vocab_size=output_vocab_size,
            dropout=dropout
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _inject_struct_features(self, h_local, row_sum, entropy, mask=None):
        if row_sum is None or entropy is None:
            return h_local

        struct_feat = torch.stack([row_sum, entropy], dim=-1)  # (B, L, 2)
        struct_delta = self.struct_proj(struct_feat)
        if mask is not None:
            struct_delta = struct_delta * mask.unsqueeze(-1).float()
        return h_local + struct_delta

    def _make_dummy_router_outputs(self, batch_size, seq_len, mask, device, dtype):
        """Produce placeholder router outputs for no_chunk mode so that loss
        functions and diagnostics that expect these keys keep working without
        a separate code path. boundary_probs is all-zeros, expected_segments=1,
        and segment_ids assigns every token to chunk 0."""
        dummy_boundary = torch.zeros(
            batch_size, max(seq_len - 1, 1), device=device, dtype=dtype
        )
        segment_ids = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
        if mask is not None:
            segment_ids = segment_ids.masked_fill(mask == 0, -1)
        return {
            "boundary_mask": dummy_boundary,
            "boundary_logits": dummy_boundary,
            "boundary_probs": dummy_boundary,
            "segment_ids": segment_ids,
            "expected_segments": torch.ones(batch_size, device=device, dtype=dtype),
        }

    def forward(self, x, row_sum=None, entropy=None, cross_pair_sum=None, mask=None):
        batch_size, seq_len = x.shape
        device = x.device

        # 1. Local encoding (shared)
        h_local = self.local_encoder(x, mask=mask)

        if self.use_struct_injection:
            h_local_struct = self._inject_struct_features(h_local, row_sum, entropy, mask)
        else:
            h_local_struct = h_local
        # 2. Chunking decision (optional)
        if self.use_no_chunk:
            chunk_repr = h_local_struct
            if mask is not None:
                src_key_padding_mask = (mask == 0)
            else:
                src_key_padding_mask = torch.zeros(
                    batch_size, seq_len, device=device, dtype=torch.bool
                )
            router_outputs = self._make_dummy_router_outputs(
                batch_size, seq_len, mask, device, h_local.dtype
            )
            segment_ids_for_dechunk = None
            boundary_probs_for_dechunk = None
        else:
            router_outputs = self.dynamic_router(h_local_struct, cross_pair_sum, mask=mask)
            segment_ids = router_outputs["segment_ids"]
            boundary_probs = router_outputs["boundary_probs"]

            downsample_out = self.downsampler(h_local_struct, segment_ids, mask=mask)
            chunk_repr = downsample_out["chunk_repr"]
            src_key_padding_mask = downsample_out["src_key_padding_mask"]
            segment_ids_for_dechunk = segment_ids
            boundary_probs_for_dechunk = boundary_probs

        # 3. Shared LatentTransformer
        h_latent, src_key_padding_mask = self.latent_transformer(
            chunk_repr,
            src_key_padding_mask=src_key_padding_mask
        )

        # 4. Shared classification head (pool h_latent over valid positions)
        valid_mask = ~src_key_padding_mask
        seg_counts = valid_mask.sum(dim=1, keepdim=True).float()
        pooled = (h_latent * valid_mask.unsqueeze(-1).float()).sum(dim=1) / (seg_counts + 1e-8)
        class_logits = self.classifier(pooled)

        # 5. Upsample / dechunk (optional)
        if self.use_no_chunk:
            token_features = h_latent  # already token-level
            smoothed_chunks = h_latent
            upsampled_chunks = h_latent
            transition_probs = h_latent.new_zeros(
                (batch_size, max(seq_len - 1, 0))
            )
        else:
            bppm_row = row_sum
            if bppm_row is not None and mask is not None:
                bppm_row = bppm_row * mask.float()

            dechunk_out = self.dechunker(
                h_latent,
                h_local_struct,
                segment_ids=segment_ids_for_dechunk,
                boundary_probs=boundary_probs_for_dechunk,
                src_key_padding_mask=src_key_padding_mask,
                mask=mask,
                bppm_row=bppm_row,
            )
            token_features = dechunk_out["token_features"]
            smoothed_chunks = dechunk_out["smoothed_chunks"]
            upsampled_chunks = dechunk_out["upsampled"]
            transition_probs = dechunk_out["transition_probs"]

        # 6. Reconstruction head (shared)
        recon_logits = self.recon_decoder(token_features)

        return {
            "boundary_mask": router_outputs["boundary_mask"],
            "boundary_logits": router_outputs["boundary_logits"],
            "boundary_probs": router_outputs["boundary_probs"],
            "segment_ids": router_outputs["segment_ids"],
            "expected_segments": router_outputs["expected_segments"],
            "h_local": h_local,
            "h_local_struct": h_local_struct,
            "chunk_repr": chunk_repr,
            "h_latent": h_latent,
            "smoothed_chunks": smoothed_chunks,
            "upsampled_chunks": upsampled_chunks,
            "token_features": token_features,
            "transition_probs": transition_probs,
            "recon_logits": recon_logits,
            "src_key_padding_mask": src_key_padding_mask,
            "class_logits": class_logits,
            "pooled": pooled,
        }



