import os
import pandas as pd
import numpy as np
import shutil


names = ["blocker", "ago", "e-"]
shots_list = [10, 20, 40]
split_nums = [5, 6, 7, 8, 9]

model_path = "var_models"
pkl_path = "var_pkls"

data_source_dir = "var_data"

if not os.path.exists(data_source_dir):
    print(f"Error: Data source directory {data_source_dir} does not exist. Please run the previous script to generate merged data first.")
    exit(1)

reference_folder = "ago_10shots_0"
if os.path.exists(reference_folder):
    train_count = len(pd.read_csv(os.path.join(reference_folder, "train.tsv"), sep='\t', header=None))
    valid_count = len(pd.read_csv(os.path.join(reference_folder, "valid.tsv"), sep='\t', header=None))
    test_count = len(pd.read_csv(os.path.join(reference_folder, "test.tsv"), sep='\t', header=None))
    print(f"Sample counts in reference folder: train={train_count}, valid={valid_count}, test={test_count}")
else:
    print(f"Warning: Reference folder {reference_folder} does not exist, using default sample counts")
    train_count = 10
    valid_count = 20
    test_count = None

for name in names:
    
    
    
    
    for shots in shots_list:
        
        
        
        for split_num in split_nums:
            
            
            
            
            
            
            
            
            

            os.makedirs(f"ablation/{pkl_path}/{name}_{shots}shots_{split_num}", exist_ok=True)
            os.makedirs(f"ablation/{model_path}/{name}_{shots}shots_{split_num}", exist_ok=True)

print("Data splitting completed!") 