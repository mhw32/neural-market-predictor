# This file contains the functions used to load and edit data 
# from MATLAB before any true labelling / classification is 
# performed.

# Please see the various ipython notebook for visual context.

from scipy.io import loadmat
import numpy as np
from __future__ import division, print_function
import matplotlib.pyplot as plt
import sys
from pickle import load
sys.path.insert(1, r'./')
from detect_cusum import detect_cusum
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu
from scipy.stats.mstats import normaltest
import scipy.stats as stats
from pylab import *
from scipy.stats import ttest_1samp, mannwhitneyu

# ----------------------------------------------------------------
# MAIN PROGRAM -- TO DO THE CODE

def load_data(indexpath, datapath):
	index = load(open(indexpath, 'rb'))
	data = load(open(datapath, 'rb'))
	return index, data

def label_data(index, data, ticker):
	# Index - The feature tags themselves
	# Data  - The actual feature vectors
	ticker = search_by_ticker(data, ticker)
	stock_ticker = search_by_index(index, ticker, 'Log Stock Return')
	stock_ticker = np.array([float(i) for i in stock_ticker])
	stock_ticker = remove_nan(stock_ticker)
	index_ticker = search_by_index(index, ticker, 'Log Index Return')
	index_ticker = np.array([float(i) for i in index_ticker])
	index_ticker = remove_nan(index_ticker)

	# Do CUSUM (with some random parameters for now...)
	threshold, drift, ending, show, ax = 0.15, 0.06, True, True, None
	ta, tai, taf, amp = detect_cusum(stock_ticker, threshold, drift, ending, show, ax)
	# let's add the beginning and end index to ta
	ta = list(ta)
	if (ta[-1] != len(stock_ticker)):
	    ta.append(len(stock_ticker))
	ta = [0] + ta
	# Split the data by CUSUM
	split_stock_cusum = np.array([stock_ticker[i:j] for i,j in zip(ta[0:len(ta)-1], ta[1:])])
	split_index_cusum = np.array([index_ticker[i:j] for i,j in zip(ta[0:len(ta)-1], ta[1:])])

	# Let's work with the normality tests + label
	labels = np.array([one_tailed_two_sample_t_test(i,j) for i,j in zip(split_index_cusum, split_stock_cusum)])
	# Might be too little data...
	return labels

# ----------------------------------------------------------------
# CODE TO CONVERT DATA TO PYTHON

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

# ----------------------------------------------------------------
# CODE TO RUN CUSUM

# This function searches by ticker, i.e. 'AA'
def search_by_ticker(data, ticker):
    return np.array([i for i in data if i[0] == ticker])

# This functions searches by index i.e. Log Revenue Return
def search_by_index(index, data, spec):
    idx = np.where(index == spec)[0][0]
    return np.array([i[idx] for i in data])

# This function just recurses through possible options in small 
# steps and stores the good ones.
def searchParams(data, step, maxim):
    ending, show, ax = True, False, None
    # temporary variables
    init_thres = maxim
    init_drift = 0.0
    # storage variables
    possible_params = []
    
    while init_thres >= 0:
        while init_drift <= maxim:
            ta, tai, taf, amp = detect_cusum(stock_AA, \
                threshold, drift, ending, show, ax)
            if len(ta) > 0:
                possible_params.append(tup(init_thres, init_drift))
            
            init_drift = init_drift + step      
        init_thres = init_thres - step
        # Set the other thing back to normal
        init_drift = 0.0
    return possible_params

# ----------------------------------------------------------------
# T-TESTS TO RUN THESE

# This works because the t_stat is SIGNED. Thus if it is > 0, 
# then we can reject in favor of a greater than alternative. 
def one_tailed_one_sample_t_test(data, alpha=0.055, hypothesis="greater"):
    if hypothesis not in ["greater", "less"]:
        return "Error: No hypothesis."
    t_stat, prob = ttest_1samp(data, 0.0)
    # Depends on what the hypothesis is
    if hypothesis == "greater": # t-stat is the indicator
        result = True if (t_stat > 0 and prob / 2 < alpha) else False
    else:
        result = True if (t_stat < 0 and prob / 2 < alpha) else False
    return result

# Basically same thing but with two samples.
def one_tailed_two_sample_t_test(data1, data2, alpha=0.055, hypothesis="greater"):
    if hypothesis not in ["greater", "less"]:
        return "Error: No hypothesis."
    normality1 = am_i_normal(data1)
    normality2 = am_i_normal(data2)
    
    if normality1 and normality2:
        t_stat, prob = ttest_ind(data1, data2, equal_var=True)
        if hypothesis == "greater": # t-stat is the indicator
            result = True if (t_stat > 0 and prob / 2 < alpha) else False
        else:
            result = True if (t_stat < 0 and prob / 2 < alpha) else False
    else:
        u_stat, prob = mannwhitneyu(data1, data2)
        if prob < 0.055:
            # All I know at this point is that the distributions are distinct. 
            # I have to compare raw medians to know which way
            # Let data1 be stock data
            result = np.median(data1) > np.median(data2)  
        else:
        	result = False                          
    return result

# ----------------------------------------------------------------
# NORMAL TESTING FOR CODE

# my threshold will be 0.055
# I'm going to use a scipy normaltest statistic. 
# It will test skew and with a chi-test: 
# If the probability is small, then there is a large chance that the 
# distribution is NOT NORMAL.
def am_i_normal(data, threshold=0.055):
    chi, p_val = stats.normaltest(x)
    if p_val < threshold:
        return False
    return True

def floatify(data):
    return np.array([float(i) for i in data])

def remove_nan(data):
    return data[~np.isnan(data)]

def visual_normal_test(data):
    h = sorted(remove_nan(data))
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))
    
    title('Normality Plot for Data for Log Revenue Return')
    plot(h, fit,'-o')
    hist(h, normed=True) 
    grid(True)
    show()
