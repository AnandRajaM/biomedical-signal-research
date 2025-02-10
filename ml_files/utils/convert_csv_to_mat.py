import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat

# Load the original .mat file
mat_file_path = r'D:\24phd7039\biomedical-research\datasets\mat_files\datasets_subject_01_to_10_scidata\GDN0003\GDN0003_1_Resting.mat'  # Change to your .mat file path
mat_data = loadmat(mat_file_path)

# Load the CSV file for radar_i, radar_q, tfm_ecg1, tfm_ecg2
csv_file_path = r'D:\24phd7039\biomedical-research\ml_files\utils\plot_folder\GDN0003\combined.csv'  # Change to your CSV file path
csv_data = pd.read_csv(csv_file_path)

# Create data_dict in the exact order specified
data_dict = {
    '__header__': mat_data['__header__'],
    '__version__': mat_data['__version__'],
    '__globals__': mat_data['__globals__'],
    'fs_bp': mat_data['fs_bp'].astype(np.float64),  # Convert to double
    'fs_ecg': mat_data['fs_ecg'].astype(np.float64),  # Convert to double
    'fs_icg': mat_data['fs_icg'].astype(np.float64),  # Convert to double
    'fs_intervention': mat_data['fs_intervention'].astype(np.float64),  # Convert to double
    'fs_radar': mat_data['fs_radar'].astype(np.float64),  # Convert to double
    'fs_z0': mat_data['fs_z0'].astype(np.float64),  # Convert to double
    'measurement_info': mat_data['measurement_info'],
    'radar_i': csv_data['radar_i'].values.reshape(-1, 1).astype(np.float64) if 'radar_i' in csv_data.columns else np.array([]).reshape(-1, 1),
    'radar_q': csv_data['radar_q'].values.reshape(-1, 1).astype(np.float64) if 'radar_q' in csv_data.columns else np.array([]).reshape(-1, 1),
    'tfm_bp': mat_data['tfm_bp'],
    'tfm_ecg1': mat_data['tfm_ecg1'].astype(np.float64) if 'tfm_ecg1' in mat_data else np.array([]).reshape(-1, 1),  # Take from .mat file
    'tfm_ecg2': mat_data['tfm_ecg2'].astype(np.float64) if 'tfm_ecg2' in mat_data else np.array([]).reshape(-1, 1),#.values.reshape(-1, 1).astype(np.float64) if 'tfm_ecg2' in csv_data.columns else np.array([]).reshape(-1, 1),
    'tfm_icg': mat_data['tfm_icg'],
    'tfm_intervention': mat_data['tfm_intervention'],
    'tfm_z0': mat_data['tfm_z0'],
    'tfm_param': mat_data['tfm_param'],
    'tfm_param_time': mat_data['tfm_param_time'],
}


# Save the combined data to a new .mat file
new_mat_file_path = r'D:\24phd7039\biomedical-research\ml_files\utils\plot_folder\GDN0003\GDN0003_1_Resting.mat'  # Change to your desired output path
savemat(new_mat_file_path, data_dict)

print(f"Data has been combined and saved to {new_mat_file_path}")
