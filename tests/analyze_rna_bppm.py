import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def analyze_bppm_data():
    # 配置路径
    input_file = os.path.join(_REPO_ROOT, 'data', 'pre_random', 'rna_bppm_data.pkl')
    output_dir = os.path.join(_REPO_ROOT, 'source')
    summary_file = os.path.join(output_dir, 'RNA_BPPM_Summary.txt')
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    # 1. 加载数据
    print(f"Loading data from {input_file}...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    df = pd.DataFrame(data)
    total_count = len(df)
    
    # 2. 提取统计信息
    summary_lines = []
    header = f"RNA BPPM Data Analysis Summary\n" + "="*40 + "\n"
    summary_lines.append(header)
    summary_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary_lines.append(f"Source File: {input_file}\n")
    summary_lines.append(f"Total Sequences Processed: {total_count}\n")
    summary_lines.append("-" * 40 + "\n\n")

    # 3. 分析前 10 条数据
    print("\n" + "="*60)
    print(f"{'ID':<25} | {'Length':<6} | {'Max Prob':<10} | {'Entropy/Non-z%':<16}")
    print("-" * 60)
    
    top_10_details = "Top 10 Sequences Detailed Info:\n" + "-"*30 + "\n"
    summary_lines.append(top_10_details)

    for i in range(min(10, total_count)):
        row = df.iloc[i]

        # 检测格式
        if 'row_sum' in row:
            # 新向量格式
            row_sum = row['row_sum']
            seq_len = len(row_sum)
            max_prob = float(np.max(row_sum))
            top1_prob = row['top1_prob']
            mean_entropy = float(np.mean(row['entropy']))
            record_id = row.get('id', f'idx_{i}')

            console_line = f"{str(record_id):<25} | {seq_len:<6} | {max_prob:<10.4f} | {mean_entropy:<10.4f}"
            print(console_line)

            summary_lines.append(f"[{i+1}] ID: {record_id}\n")
            summary_lines.append(f"    Length (from row_sum): {seq_len}\n")
            summary_lines.append(f"    Max Row Sum: {max_prob:.6f}\n")
            summary_lines.append(f"    Mean Entropy: {mean_entropy:.6f}\n")
            summary_lines.append(f"    Mean Top-1 Prob: {float(np.mean(top1_prob)):.6f}\n")
            summary_lines.append("\n")
        else:
            # 旧矩阵格式
            bppm = row['bppm']
            seq_len = row['seq_len']
            valid_bppm = bppm[:seq_len, :seq_len]
            max_prob = np.max(valid_bppm)
            nonzero_count = np.count_nonzero(valid_bppm)
            nonzero_ratio = (nonzero_count / (seq_len * seq_len)) * 100

            console_line = f"{row['id']:<25} | {seq_len:<6} | {max_prob:<10.4f} | {nonzero_ratio:<10.2f}%"
            print(console_line)

            summary_lines.append(f"[{i+1}] ID: {row['id']}\n")
            summary_lines.append(f"    Sequence: {row['sequence'][:50]}...\n")
            summary_lines.append(f"    Length: {seq_len}\n")
            summary_lines.append(f"    Max Pairing Probability: {max_prob:.6f}\n")
            summary_lines.append(f"    Sparsity (Non-zero ratio): {nonzero_ratio:.2f}%\n")
            summary_lines.append("\n")

    # 4. 全局统计
    if 'seq_len' in df.columns:
        all_lens = df['seq_len'].values
    else:
        all_lens = np.array([len(row['row_sum']) for _, row in df.iterrows()])
    global_stats = (
        f"Global Dataset Statistics:\n"
        f"{'-'*30}\n"
        f"Max Sequence Length: {np.max(all_lens)}\n"
        f"Min Sequence Length: {np.min(all_lens)}\n"
        f"Average Sequence Length: {np.mean(all_lens):.2f}\n"
        f"Standard Deviation of Lengths: {np.std(all_lens):.2f}\n"
    )
    summary_lines.append(global_stats)
    
    print("="*60)
    print("\nGlobal Statistics:")
    print(f"Max Length: {np.max(all_lens)}")
    print(f"Avg Length: {np.mean(all_lens):.2f}")

    # 5. 保存到文本文件
    print(f"\nSaving summary to {summary_file}...")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)
    
    print("Analysis complete.")

if __name__ == "__main__":
    # 确保在 conda 环境中运行所需的 pandas/numpy 已安装
    analyze_bppm_data()
