import os
import pandas as pd

input_base_path = r'D:\24phd7039\biomedical-research\datasets\csv_files'
output_base_path = r'D:\24phd7039\biomedical-research\datasets\normalized_csv'

def normalize_data(df):
    return (df - df.mean()) / df.std()

for root, dirs, files in os.walk(input_base_path):
    for file in files:
        if file.endswith('.csv'):
            input_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_base_path)
            output_dir = os.path.join(output_base_path, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            
            output_file_path = os.path.join(output_dir, file)
            
            # Load the CSV file
            df = pd.read_csv(input_file_path)
            
            # Normalize each column
            normalized_df = df.apply(normalize_data)
            
            # Save to the output path
            normalized_df.to_csv(output_file_path, index=False)

            print(f'Normalized {file} and saved to {output_file_path}')
