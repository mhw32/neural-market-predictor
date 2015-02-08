# This file contains the functions used to setup params
# for CUSUM algorithm.

# Please see ipython notebook for visual context.

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, r'./')
from detect_cusum import detect_cusum

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