import pickle
import numpy as np
import os
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_top3_rna():
    input_path = os.path.join(_REPO_ROOT, 'data', 'pre_random', 'rna_bppm_data.pkl')
    output_dir = os.path.join(_REPO_ROOT, 'source')
    output_path = os.path.join(output_dir, 'rna_top3_bppm.xlsx')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {input_path}...")
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    records = []
    if isinstance(data, list):
        records = data[:3]
    elif isinstance(data, dict):
        records = [v for k, v in list(data.items())[:3]]
    else:
        print(f"Unknown data structure: {type(data)}")
        return

    print(f"Saving BPPM matrices to {output_path}...")
    
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            saved_count = 0
            for i, record in enumerate(records):
                if isinstance(record, dict) and 'row_sum' in record:
                    # 新向量格式：每个向量一个 sheet
                    print(f"Record #{i+1}: New vector format, exporting vectors")
                    for vec_name in ("row_sum", "entropy", "top1_partner", "top1_prob", "cross_pair"):
                        vec = record[vec_name]
                        if isinstance(vec, np.ndarray):
                            df = pd.DataFrame(vec, columns=[vec_name])
                            df.to_excel(writer, sheet_name=f"R{i+1}_{vec_name}", index=False)
                    saved_count += 1
                elif isinstance(record, dict) and 'bppm' in record:
                    # 旧矩阵格式
                    bppm = record['bppm']
                    if isinstance(bppm, np.ndarray):
                        print(f"Record #{i+1}: Found BPPM matrix with shape {bppm.shape}")
                        df = pd.DataFrame(bppm)
                        df.to_excel(writer, sheet_name=f"RNA_{i+1}", header=False, index=False)
                        saved_count += 1
                else:
                    # 尝试找 2D 矩阵
                    bppm = None
                    if isinstance(record, dict):
                        for k, v in record.items():
                            if isinstance(v, np.ndarray) and v.ndim == 2:
                                bppm = v
                                break
                    if bppm is not None:
                        print(f"Record #{i+1}: Found BPPM matrix with shape {bppm.shape}")
                        df = pd.DataFrame(bppm)
                        df.to_excel(writer, sheet_name=f"RNA_{i+1}", header=False, index=False)
                        saved_count += 1
                    else:
                        print(f"Record #{i+1}: No valid data found.")
            
            if saved_count > 0:
                print(f"Successfully saved {saved_count} matrices to {output_path}")
            else:
                print("No matrices were saved.")
                
    except Exception as e:
        print(f"Error saving to Excel: {e}")

if __name__ == "__main__":
    extract_top3_rna()
