import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


input_base_path = r'D:\dev\biomedical-signal-research\matlab_files\mat datasets'
output_base_path = r'D:\dev\biomedical-signal-research\ml_files\datasets'


dataset_folders = [
    'datasets_subject_1_to_10_scidata',
    'datasets_subject_11_to_20_scidata',
    'datasets_subject_21_to_30_scidata'
]

for dataset_folder in dataset_folders:
    input_folder = os.path.join(input_base_path, dataset_folder)
    output_folder = os.path.join(output_base_path, dataset_folder)

    os.makedirs(output_folder, exist_ok=True)
    
    for gdn_folder_name in os.listdir(input_folder):
        input_gdn_folder = os.path.join(input_folder, gdn_folder_name)
        output_gdn_folder = os.path.join(output_folder, gdn_folder_name)
        
        os.makedirs(output_gdn_folder, exist_ok=True)
        
        for mat_file in os.listdir(input_gdn_folder):
            if mat_file.endswith('.mat'):
                mat_file_path = os.path.join(input_gdn_folder, mat_file)

                mat_data = loadmat(mat_file_path)

                radar_i = mat_data['radar_i'].flatten() if 'radar_i' in mat_data else np.array([])
                radar_q = mat_data['radar_q'].flatten() if 'radar_q' in mat_data else np.array([])
                ecg2 = mat_data['tfm_ecg2'].flatten() if 'tfm_ecg2' in mat_data else np.array([])

                df = pd.DataFrame({
                    'radar_i': radar_i,
                    'radar_q': radar_q,
                    'tfm_ecg2': ecg2
                })

                csv_file_name = mat_file.replace('.mat', '.csv')
                output_csv_file = os.path.join(output_gdn_folder, csv_file_name)

                df.to_csv(output_csv_file, index=False)

                print(f'Successfully converted {mat_file} to csv')
