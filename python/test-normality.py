# This file contains the functions used to check for normality
# before applying any t-tests for labelling.

# Please see ipython notebook for visual context.

from scipy.stats.mstats import normaltest
import scipy.stats as stats
from pylab import *
import numpy as np

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