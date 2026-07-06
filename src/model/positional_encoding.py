import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    标准正弦位置编码 (Sinusoidal Positional Encoding)
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model (int): 特征维度
            max_len (int): 最大序列长度
            dropout (float): Dropout 概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 Batch 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # register_buffer 确保 pe 不会被视为模型参数，但会随模型移动到 GPU
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征，形状为 (Batch, Seq_len, Dim)
        Returns:
            torch.Tensor: 加上位置编码后的特征
        """
        # x.size(1) 是当前序列的实际长度
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
