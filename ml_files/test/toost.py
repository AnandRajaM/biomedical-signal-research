from scipy.io import loadmat

# Load the .mat file
mat_data = loadmat(r'')

# List all variables in the .mat file
variables = mat_data.keys()
print(variables)
