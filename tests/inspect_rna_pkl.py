import pickle
import os
import numpy as np
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def inspect_rna_pkl():
    file_path = os.path.join(_REPO_ROOT, 'data', 'pre_random', 'rna_bppm_data.pkl')
    
    # 1. 安全加载文件
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"异常: 无法加载 pickle 文件。错误信息: {e}")
        return

    print(f"已加载文件: {file_path}")
    print(f"数据顶层类型: {type(data)}")

    # 准备遍历的数据迭代器
    # 兼容处理：虽然题目假设是 dict，但实际数据可能是 list of dicts，这里做自动适配
    iterator = []
    
    if isinstance(data, dict):
        # 符合题目假设的结构：Key -> Matrix
        print("检测到字典结构，按键值对处理...")
        iterator = data.items()
    elif isinstance(data, list):
        # 实际可能存在的结构：List of Dicts (常见于数据集)
        print("检测到列表结构，尝试提取信息...")
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                seq_id = item.get('id', str(idx))
                # 新格式：向量字典
                if 'row_sum' in item:
                    vec_info = {k: (v.shape if isinstance(v, np.ndarray) else type(v).__name__)
                                for k, v in item.items()}
                    iterator.append((seq_id, vec_info))
                # 旧格式：完整矩阵
                else:
                    matrix = item.get('bppm')
                    if matrix is None:
                        for v in item.values():
                            if isinstance(v, np.ndarray) and v.ndim == 2:
                                matrix = v
                                break
                    iterator.append((seq_id, matrix))
            else:
                iterator.append((str(idx), item))
    else:
        print("警告: 数据结构既不是字典也不是列表，无法自动解析。")
        return

    # 统计变量
    valid_count = 0
    dimensions = [] # 存储 (rows, cols)

    print("\n" + "-"*30)
    
    # 3. 逐条读取并处理
    for seq_id, content in iterator:
        # 新格式：向量字典
        if isinstance(content, dict):
            valid_count += 1
            shapes_str = ", ".join(f"{k}={v}" for k, v in content.items())
            print(f"序列ID: {seq_id}, 向量格式: {{{shapes_str}}}")
            continue

        matrix = content
        # 尝试转换为 numpy array (如果不是的话)
        if not isinstance(matrix, np.ndarray):
            try:
                matrix = np.array(matrix)
            except:
                print(f"警告: 序列ID {seq_id} 的内容无法转换为 NumPy 数组，跳过。")
                continue

        # 检查维度
        if matrix.ndim != 2:
            print(f"警告: 序列ID {seq_id} 的数据不是二维矩阵 (Shape: {matrix.shape})，跳过。")
            continue

        rows, cols = matrix.shape
        dimensions.append((rows, cols))
        valid_count += 1

        # 打印信息
        print(f"序列ID: {seq_id}, 矩阵维度: {rows}×{cols}")

    # 5. 打印统计摘要
    print("-" * 30)
    print("统计摘要")
    print("-" * 30)
    print(f"总有效序列条数: {valid_count}")

    if valid_count > 0:
        rows_list = [d[0] for d in dimensions]
        cols_list = [d[1] for d in dimensions]
        
        min_dim = (min(rows_list), min(cols_list))
        max_dim = (max(rows_list), max(cols_list))
        avg_dim = (np.mean(rows_list), np.mean(cols_list))
        
        print(f"最小维度: {min_dim[0]}×{min_dim[1]}")
        print(f"最大维度: {max_dim[0]}×{max_dim[1]}")
        print(f"平均维度: {avg_dim[0]:.2f}×{avg_dim[1]:.2f}")
    else:
        print("无有效统计数据。")

if __name__ == "__main__":
    inspect_rna_pkl()
