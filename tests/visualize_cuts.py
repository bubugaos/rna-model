import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import RNADynamicModel

def generate_mock_data(seq_len=100):
    """
    生成模拟的 RNA 序列和具有结构特征的 BPPM 矩阵。
    结构特征：在某些区域有明显的对角线（茎区），在其他区域则为空（环区）。
    """
    # 模拟序列 (A, U, C, G) -> (0, 1, 2, 3), Padding -> 4
    x = torch.randint(0, 4, (1, seq_len))
    
    # 模拟 BPPM: 主要是对角线附近的概率
    bppm = torch.zeros((1, seq_len, seq_len))
    
    # 模拟两个茎区 (Stems)
    # Stem 1: positions 10-20 pairs with 40-50
    for i in range(10):
        bppm[0, 10+i, 50-i] = 0.9
        bppm[0, 50-i, 10+i] = 0.9
        
    # Stem 2: positions 60-70 pairs with 85-95
    for i in range(10):
        bppm[0, 60+i, 95-i] = 0.8
        bppm[0, 95-i, 60+i] = 0.8
    
    # 添加一些噪声
    bppm += torch.rand((1, seq_len, seq_len)) * 0.05
    bppm = torch.clamp(bppm, 0, 1)
    
    return x, bppm

def load_real_sample(file_path, sample_idx=0):
    """加载真实的数据样本"""
    if not os.path.exists(file_path):
        return None, None
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list) and len(data) > sample_idx:
        item = data[sample_idx]
        if isinstance(item, dict):
            # 旧格式：完整 BPPM 矩阵
            bppm = item.get('bppm')
            if bppm is not None:
                bppm_tensor = torch.from_numpy(bppm).float().unsqueeze(0)
                seq_len = bppm_tensor.size(1)
                x = torch.randint(0, 4, (1, seq_len))
                return x, bppm_tensor
            # 新向量格式：无法重建完整矩阵用于可视化
            if 'row_sum' in item:
                print("注意: 数据为新向量格式，无法重建完整 BPPM 矩阵进行可视化。")
                print("请使用旧格式 pkl 文件，或使用 mock 数据。")
            
    return None, None

def visualize_cuts(model_path=None, data_path=None, output_file="rna_cuts_visualization.png", max_len=200):
    # 1. 准备数据
    seq_len = 100
    if data_path:
        x, bppm = load_real_sample(data_path)
        if x is None:
            print(f"无法从 {data_path} 加载数据，使用模拟数据。")
            x, bppm = generate_mock_data(seq_len)
        else:
            # 如果序列太长，截断以便可视化
            if x.size(1) > max_len:
                print(f"原始序列长度 {x.size(1)} 超过 {max_len}，正在截断以便可视化...")
                x = x[:, :max_len]
                bppm = bppm[:, :max_len, :max_len]
            seq_len = x.size(1)
            print(f"成功加载真实数据，可视化序列长度: {seq_len}")
    else:
        x, bppm = generate_mock_data(seq_len)
        print(f"使用模拟数据，序列长度: {seq_len}")

    # 2. 初始化/加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNADynamicModel(vocab_size=5, embed_dim=128, nhead=4, num_layers=2).to(device)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"已加载模型权重: {model_path}")
    else:
        print("未指定模型权重或文件不存在，使用初始化模型。")
    
    model.eval()
    
    # 3. 模型前向传播获取边界
    with torch.no_grad():
        x = x.to(device)
        bppm = bppm.to(device)
        # Extract vectors from BPPM for the new model interface
        row_sum = bppm.sum(dim=-1)
        p_safe = bppm.clamp(min=1e-8)
        entropy = -(p_safe * p_safe.log()).sum(dim=-1) / float(bppm.size(1))
        S = bppm.cumsum(dim=1).cumsum(dim=2)
        cross_pair_sum = (S[:, :, -1] - torch.diagonal(S, dim1=1, dim2=2))[:, :-1]
        outputs = model(x, row_sum=row_sum, entropy=entropy, cross_pair_sum=cross_pair_sum)
        boundary_mask = outputs["boundary_mask"][0].cpu().numpy() # (L-1,)
    
    # 4. 绘图
    plt.figure(figsize=(12, 10))
    
    # 绘制 BPPM 热力图
    # 我们主要看上三角部分，因为是对称的
    bppm_np = bppm[0].cpu().numpy()
    
    # 使用 seaborn 绘制热力图
    ax = sns.heatmap(bppm_np, cmap="YlGnBu", cbar_kws={'label': 'Pairing Probability'})
    
    # 找出切分边界 (boundary_mask > 0.5)
    # boundary_mask[i] 对应于位置 i 和 i+1 之间的边界
    cut_indices = np.where(boundary_mask > 0.5)[0]
    
    # 在热力图上绘制垂直虚线
    # 边界 i 位于碱基 i 和 i+1 之间，在坐标轴上大约是 i + 1 的位置
    for idx in cut_indices:
        plt.axvline(x=idx + 1, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    plt.title("RNA Sequence Chunking Visualization\n(Dashed lines indicate model-predicted boundaries)", fontsize=15)
    plt.xlabel("Sequence Position", fontsize=12)
    plt.ylabel("Sequence Position", fontsize=12)
    
    # 添加图例说明
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', linestyle='--', lw=2)]
    plt.legend(custom_lines, ['Predicted Boundary'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"可视化结果已保存至: {output_file}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize RNA chunking boundaries on BPPM heatmap.")
    parser.add_argument("--data", type=str, default=os.path.join("data", "pre_random", "rna_bppm_data.pkl"), help="Path to BPPM pickle file")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="rna_cuts_visualization.png", help="Output image path")
    parser.add_argument("--max_len", type=int, default=200, help="Maximum sequence length to visualize")
    parser.add_argument("--mock", action="store_true", help="Force use mock data")
    
    args = parser.parse_args()
    
    if args.mock:
        visualize_cuts(model_path=args.model, data_path=None, output_file=args.output, max_len=args.max_len)
    else:
        visualize_cuts(model_path=args.model, data_path=args.data, output_file=args.output, max_len=args.max_len)
