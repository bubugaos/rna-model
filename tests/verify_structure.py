import pickle
import os
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(_REPO_ROOT, 'data', 'pre_random', 'rna_bppm_data.pkl')

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("-" * 30)
    print(f"Top level type: {type(data)}")
    
    if isinstance(data, list):
        print(f"Is List: Yes")
        print(f"List length: {len(data)}")
        if len(data) > 0:
            first_item = data[0]
            print(f"Item [0] type: {type(first_item)}")
            
            if isinstance(first_item, dict):
                print(f"Is Item Dict: Yes")
                keys = list(first_item.keys())
                print(f"Item keys: {keys}")
                print(f"Has 'bppm' key (old format): {'bppm' in first_item}")
                print(f"Has 'row_sum' key (new vector format): {'row_sum' in first_item}")
                if 'row_sum' in first_item:
                    for k in ("row_sum", "entropy", "top1_partner", "top1_prob", "cross_pair"):
                        v = first_item.get(k)
                        shape = v.shape if isinstance(v, np.ndarray) else "N/A"
                        print(f"  {k}: shape={shape}, dtype={v.dtype if isinstance(v, np.ndarray) else 'N/A'}")
            else:
                print(f"Is Item Dict: No")
    elif isinstance(data, dict):
        print(f"Is List: No (It is a Dict)")
        # Check values
        if len(data) > 0:
            first_val = next(iter(data.values()))
            print(f"First value type: {type(first_val)}")
            if isinstance(first_val, dict):
                 print(f"First value keys: {list(first_val.keys())}")
                 print(f"Has 'bppm' key (old format): {'bppm' in first_val}")
                 print(f"Has 'row_sum' key (new vector format): {'row_sum' in first_val}")
    else:
        print(f"Is List: No (Type: {type(data)})")
        
    print("-" * 30)

except Exception as e:
    print(f"Error: {e}")
