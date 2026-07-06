import torch.nn as nn
from .positional_encoding import PositionalEncoding
from src.config import RNA_INPUT_VOCAB_SIZE

class LocalEncoder(nn.Module):
    """
    Local Encoder
    
    Uses nn.Embedding and nn.TransformerEncoder for local feature extraction.
    Includes PositionalEncoding.
    
    Args:
        vocab_size (int): Vocabulary size. Default 7 (A, U, C, G, PAD, MASK, UNK).
        embed_dim (int): Embedding dimension. Default 256.
        nhead (int): Transformer heads. Default 4.
        num_layers (int): Transformer layers. Default 2.
        dim_feedforward (int): Feedforward dimension. Default 512.
        dropout (float): Dropout probability.
    """
    def __init__(self, vocab_size=RNA_INPUT_VOCAB_SIZE, embed_dim=256, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(LocalEncoder, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        # 3. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input sequence Token IDs (Batch, L).
            mask (torch.Tensor, optional): Sequence mask (Batch, L), 1 for valid, 0 for padding.
            
        Returns:
            h_local (torch.Tensor): Local encoded features (Batch, L, embed_dim).
        """
        # Generate padding mask for Transformer
        # PyTorch Transformer expects src_key_padding_mask where True indicates padding (ignored)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)
            
        # 1. Embedding
        h_embedded = self.embedding(x) # (Batch, L, embed_dim)
        
        # 2. Add Positional Encoding
        h_embedded = self.pos_encoding(h_embedded)
        
        # 3. Transformer Encoding
        h_local = self.transformer_encoder(h_embedded, src_key_padding_mask=src_key_padding_mask)
        
        return h_local
