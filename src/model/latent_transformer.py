import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding


class LatentTransformer(nn.Module):
    """Process compressed chunk sequences with a standard Transformer encoder."""

    def __init__(self, dim, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(LatentTransformer, self).__init__()
        self.pos_encoding = PositionalEncoding(dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        self.dim = dim

    def forward(self, chunk_repr, src_key_padding_mask=None):
        """Encode chunk-level representations."""
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                chunk_repr.size(0),
                chunk_repr.size(1),
                device=chunk_repr.device,
                dtype=torch.bool,
            )

        h_latent = self.pos_encoding(chunk_repr)
        h_latent = h_latent.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        h_latent = self.transformer_encoder(h_latent, src_key_padding_mask=src_key_padding_mask)
        h_latent = h_latent.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return h_latent, src_key_padding_mask


