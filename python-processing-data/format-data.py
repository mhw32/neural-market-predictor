# This file contains the functions used to load and edit data 
# from MATLAB before any true labelling / classification is 
# performed.

from scipy.io import loadmat
import numpy as np

# Read a .MAT file
def mat2py(filepath, obj):
	# Load mat from matlab file
	data = loadmat(filepath)
	raw_feature_index   = data[obj][:, 0][0][0]
	raw_feature_data    = data[obj][:, 1][0]
	raw_descript_index  = data[obj][:, 2][0][0]
	raw_descript_data   = data[obj][:, 3][0]
	# Process the variables into readable Python vars
	descript_index = pythonify_stringify(raw_descript_index)
	feature_index  = pythonify_stringify(raw_feature_index)
	index = np.concatenate((descript_index, feature_index), axis=0)
	# Combine them 
	descript_data = pythonify_data(raw_descript_data)
	data = combinify(descript_data, raw_feature_data)

	return index, data

# Functiont to convert matlab unicode strings to python native 
def pythonify_stringify(matstr):
    return np.array([str(i[0]) for i in matstr])

# Function to do the same for a data object
def pythonify_data(raw_data):
    data = []
    for i in raw_data:
        data.append(np.array([str(j[0]) for j in i]))
    return np.array(data)

# Combine the overlapping data structures.
def combinify(descriptions, features):
    return np.array([np.concatenate((i, j), axis=0) \
    	for i, j in zip(descriptions, features)])