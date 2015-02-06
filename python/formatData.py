# This file contains the functions used to load and edit data 
# from MATLAB before any true labelling / classification is 
# performed.

from scipy.io import loadmat

# Read a .MAT file
def mat2py(filepath, obj):
	data = loadmat(filepath)[obj]
	return data


