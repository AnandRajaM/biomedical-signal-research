from scipy.io import loadmat

# Load the .mat file
mat_data = loadmat(r'D:\dev\biomedical-signal-research\datasets\mat_files\datasets_subject_01_to_10_scidata\GDN0001\GDN0001_1_Resting.mat')

# List all variables in the .mat file
variables = mat_data.keys()
print(variables)
