import torch
import torch.nn as nn
from src.config import RNA_OUTPUT_VOCAB_SIZE

class TaskDecoder(nn.Module):
    """
    任务解码器 (Task Decoder)
    
    一个简单的多层感知机 (MLP)，用于将潜在表示转换为最终的预测输出。
    """
    def __init__(self, latent_dim=256, hidden_dim=512, output_dim=5):
        super(TaskDecoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, h_latent):
        """
        h_latent: 潜在表示 (Batch, L_latent, latent_dim)
        返回: 预测输出 (Batch, L_latent, output_dim)
        """
        output = self.mlp(h_latent)
        return output

class ReconDecoder(nn.Module):
    """Decode fused token-level features into vocabulary logits."""

    def __init__(self, dim, vocab_size=RNA_OUTPUT_VOCAB_SIZE, dropout=0.1):
        super(ReconDecoder, self).__init__()
        self.token_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, vocab_size)
        )

    def forward(self, token_features):
        return self.token_head(token_features)


