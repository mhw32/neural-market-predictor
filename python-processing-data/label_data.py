# This file contains the functions used to load and edit data 
# from MATLAB before any true labelling / classification is 
# performed.

# Please see the various ipython notebook for visual context.

from __future__ import division, print_function
from scipy.io import loadmat
import numpy as np
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

# Define Global Errors
ERROR_MAGNITUDE = 1111
ERROR_EXISTENCE = 2222

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
# REMOVING NAN AND INF (INITIAL FEATURE PRUNING)

def remove_bad(index, data, thres, plotting=False):
    bad_counts = []
    start = 3 # add 3 because that's where we started
    end   = 124
    # we ignore the first 3 and the last 2 because the first 3 = descriptive measures, and last 2 = return data
    valid_segment = np.array(range(start, end))
    for i in valid_segment:
        feature       = data[:, i].astype(np.float32)
        nan_fraction  = np.sum(np.isnan(feature)) / float(len(feature)) * 100
        inf_fraction  = np.sum(np.isinf(feature)) / float(len(feature)) * 100
        bad_counts.append(nan_fraction + inf_fraction)
        
    bad_indices = [i for i in range(len(bad_counts)) if bad_counts[i] > thres]
    bad_indices = [i+start for i in bad_indices]
    
    # Remove the indices found
    for i in range(len(bad_indices)):
        index = np.delete(index, bad_indices[i], 0)
        data  = np.delete(data,  bad_indices[i], 1)
        for j in range(len(bad_indices)):
            bad_indices[j] = bad_indices[j]-1
    
    if plotting == True:
        title("Percentage of NaN/Inf's against Feature Number")
        plot(bad_counts)
        grid(True)
        show()
    
    return index, data

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
def test_all_params(data, step, maxim):
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

# Given reasonable constraints -- find the settings that are the most reasonable 
def search_best_params(stock_data, disp=0.20, plotting=False):
    combos = []
    # Try all the i, j combos for (threshold, drift)
    for i in np.linspace(0, disp, 20):
        for j in np.linspace(0, disp, 20):
            ta, tai, taf, amp = detect_cusum(stock_data, i, j, True, plotting, None)
            seg = [ta[k+1]-ta[k] for k in range(len(ta)-1)]
            # Let's try to pick out the ones with definitely more than 10
            seg = np.array(seg)
            # Make sure that there are no super small differenes, and none of them are 0
            if len(seg) >= 70:
                continue
            # print('threshold: %f & drift: %f' % (i, j))
            combos.append((i, j, ta, seg))
    if len(combos) > 0:
        # Sort the combos by length and return values
        combos.sort(key = lambda s: sum(s[3]))
        chosen_threshold, chosen_drift, chosen_cuts, _ = combos[::-1][0]
        chosen_cuts = cusum_consolidate(combos[::-1][0])
    else:
        if disp < 5:
            print('Unable to find anything: Moving to displacement ' + str(disp+0.05))
            chosen_threshold, chosen_drift, chosen_cuts = search_best_params(stock_data, disp+0.05, False)
        else:
            print('Returning entire segment. Completely unable to find anything.')
            chosen_threshold, chosen_drift, chosen_cuts = None, None, ERROR_EXISTENCE

    return chosen_threshold, chosen_drift, chosen_cuts

# The goal now is to select 50, and only pick the ones out of there that are > 20 
# and consolidate the rest!
def cusum_consolidate(complete):
    joined = []
    indices, shifts = complete[2], complete[3]
    base = indices[0]
    for i in range(len(indices)-1):
        if indices[i] - base > 20:
            joined.append(indices[i])
            base = indices[i]
    joined = np.array(joined)
    return joined

# This is intended to be a function to categorize the data into 3 labels
# Small, Medium, Large intervals!!! (For more comparability)
CUSUM_LABEL_CUTOFF = 45
def cusum_threshold(complete):
    small  = [] # < 45
    large  = [] # > 40
    
    indices, shifts = complete[2], complete[3]
    base = indices[0]
    for i in range(len(indices)-1):
        step = indices[i] - base
        if  step < CUSUM_LABEL_CUTOFF:
            small.append(indices[i])
        else:
            large.append(indices[i])
    small = np.array(small)
    large = np.array(large)
    return small, large

# Do the physical splitting once you have the right parameters
def split_by_cusum(data, index, ticker, plotting=False):
    #print('Starting CUSUM Evaluation')
    #print('-------------------------')
    # Pick out the stock values through ticker
    tix = search_by_ticker(data, ticker)
    stock_tix = search_by_index(index, tix, 'Log Stock Return').astype('float32')
    market_tix = search_by_index(index, tix, 'Log Index Return').astype('float32')

    if len(stock_tix) < 100:
        return None, None, ERROR_MAGNITUDE
    # Apply the CUSUM
    #print ('Finding Optimal Parameter...')
    threshold, drift, ta = search_best_params(stock_tix)
    if type(ta) == int:
        return None, None, ERROR_EXISTENCE
    #print('Set threshold: %f & drift: %f...' % (threshold, drift))
    # Do the splitting
    split_by_cusum = np.array([stock_tix[i:j] for i,j in zip(ta[0:len(ta)-1], ta[1:])])
    split_by_market = np.array([market_tix[i:j] for i,j in zip(ta[0:len(ta)-1], ta[1:])])
    #print ('Complete: time horizons calculated and split successfully...')
    return split_by_cusum, split_by_market, ta

# ----------------------------------------------------------------
# T-TESTS TO RUN THESE

def get_min_sample_size(market_data, start, end):
    # index_return = index_price_end/index_price_start (of investment horizon)
    market_return = market_data[end] / float(market_data[start])
    sigma = np.std(market_data[start:end])
    # sigma = sqrt(variance) = standard deviation
    tau = log((0.05 / 52 + market_return) / market_return) 
    n = (2.48 * sigma / float(tau))**2
    return n

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
def one_tailed_two_sample_t_test(data1, data2, alpha=0.05):
    normality1 = am_i_normal(data1)
    normality2 = am_i_normal(data2)
    
    if normality1 and normality2:
        t_stat, prob = ttest_ind(data1, data2, equal_var=True)
        # Greater than hypothesis
        if (t_stat > 0 and prob / 2 < alpha):
            result = 1
        else:
            result = 0 # Not it.
    else:
        u_stat, prob = mannwhitneyu(data1, data2)
        if prob < alpha:
            # All I know at this point is that the distributions are distinct. 
            # I have to compare raw medians to know which way
            # Let data1 be stock data
            result = (np.median(data1) > np.median(data2))
        else:
        	result = 0                          
    return result

# ----------------------------------------------------------------
# NORMAL TESTING FOR CODE

# my threshold will be 0.055
# I'm going to use a scipy normaltest statistic. 
# It will test skew and with a chi-test: 
# If the probability is small, then there is a large chance that the 
# distribution is NOT NORMAL.
def am_i_normal(data, threshold=0.055):
    chi, p_val = stats.normaltest(data)
    if p_val < threshold:
        return False
    return True

def floatify(data):
    return np.array([float(i) for i in data])

# Instant removeal of NaN = (of the NaN's I do keep)
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

# ----------------------------------------------------------------
# WRAPPER FUNCTION FOR LABELS FOR AN INDIVIDUAL STOCK

def wrapper_label_stock(index, data, tix):
    tix_data = search_by_ticker(data, tix)
    stock_tix = search_by_index(index, tix_data, 'Log Stock Return')
    index_tix = search_by_index(index, tix_data, 'Log Index Return')
    
    splitStock, splitIndex, cracks = split_by_cusum(stock_tix, index_tix, tix)
    labels = []
    for i in range(len(splitStock)):
        labels.append(tuple((cracks[i], int(one_tailed_two_sample_t_test(splitStock[i], \
            splitIndex[i], alpha=0.05)))))
    return labels


def wrapper_label_full_stock(data, index, tix, style='standard'):
    if style not in ['recursive', 'standard']:
        return 'Error. Unknown style provided.'
    
    start = 1
    split_stock, split_market, cracks = split_by_cusum(data, index, tix, False)
    part = search_by_ticker(data, tix)
    stock_tix = search_by_index(index, part, 'Log Stock Return').astype('float32')
    market_tix = search_by_index(index, part, 'Log Index Return').astype('float32')
    labels = []

    if len(stock_tix) < 100 or (type(cracks) == int and cracks == ERROR_MAGNITUDE):
        print("Error: not enough data for ticker %s." % str(tix))
        return None
    elif type(cracks) == int and cracks == ERROR_EXISTENCE:
        # Something broke... 
        print('Treating entire time data as segment for stock %s.' % str(tix))
        # Run this on the entire set
        tmp = tuple((0, int(one_tailed_two_sample_t_test(stock_tix, \
                    market_tix, alpha=0.05))))
        labels.append(tmp)
        return labels 

    for i in range(len(cracks)-1):
        start = cracks[i]
        end   = cracks[i+1]
        if style == 'recursive':
            while (end-start) >= 20:
                tmp = tuple((start, int(one_tailed_two_sample_t_test(stock_tix[start:end], \
                    market_tix[start:end], alpha=0.05))))
                labels.append(tmp)
                start += 1
        else:
            tmp = tuple((start, int(one_tailed_two_sample_t_test(stock_tix[start:end], \
                market_tix[start:end], alpha=0.05))))
            labels.append(tmp)
            start += 1
            
    return labels

def cherry_pick_all_labelled_vector(tickers, vectors_labels, data):
    complete = []
    count = 0
    for tix,vec in zip(tickers, vectors_labels):
        if vec != None:
            # Get all the data
            part = search_by_ticker(data, tix)
            # Map the data into feature-vector:label form
            tmp = [tuple((part[i][3:].astype('float32'), j)) for i,j in vec]
            complete = complete + tmp
            count = count+1
    return np.array(complete)

# ----------------------------------------------------------------
# FEATURE SELECTION MODELS

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Recursively removes attributes and builds a model with those 
# attributes that remain. Uses accuracy to identify which 
# attributes work to predicting the target attribute.
def recursive_selection(vectors, labels, num):
    model = LogisticRegression() 
    rfe   = RFE(model, num)
    rfe   = rfe.fit(vectors, labels)
    return rfe.support_

# Basically quantifies the features to see which ones 
# are "more important"
def extra_trees(vectors, labels):
    # fit an Extra Trees model to the data
    model = ExtraTreesClassifier()
    model.fit(vectors, labels)
    # display the relative importance of each attribute
    return model.feature_importances_


# Univariate feature selection works by selecting the 
# best features based on univariate statistical tests.
def chi2_selection(vectors, labels, num):
    if num >= vectors.shape[1]:
        return 'Error: features selected must be less than total number of features'
    X_new = SelectKBest(chi2, k=num).fit_transform(vectors, labels)
    return X_new

# FEATURE NUMBER SELECTION 

# Uses the RFE to do cross validation to pick the optimal # of features
def rfe_cross_validate(X, y):
    # Create the RFE object and compute a cross-validated score.
    model = LogisticRegression() 
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(y, 2), 
                  scoring='accuracy')
    rfecv.fit(X, y)
    # plot it 
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

# We can attempt a cross val method with extra trees as well.
# Let's be conservative: get the number we should use for RFE:x, return x + 5 features.
def extra_trees_cross_validate(X, y, pad):
    RFE_num = rfe_cross_validate(X, y)
    return RFE_num + int(pad)

# Given some list of chosen features, actually pick them out
def prune_vectors_given_features(feature_vectors, chosen_features, index):
    found = [np.where(index == i)[0][0] for i in chosen_features]
    found = [i-3 for i in found] # Because I look out the dates + quarter
    # Pick out from vectors
    pruned = []
    for vec in feature_vectors:
        pruned.append(np.array([vec[i] for i in found]))
    pruned = np.array(pruned)
    return pruned

# ----------------------------------------------------------------
# FULL STEPS FROM REMOVING NAN --> FEATURE SELECTION

def create_labeled_vectors(vectors_labels):
    vectors, labels = [], []
    for i in range(len(vectors_labels)):
        print('---------- Starting iteration %d -----------' % (i))
        raw = vectors_labels[i]
        tick = tickers[i]
        stock = search_by_ticker(data, tick)
        for (j,k) in raw:
            vectors.append(stock[j])
            labels.append(k)
    vectors = np.array(vectors)
    labels  = np.array(labels)
    return vectors, labels

def floatify_vector(vector):
    return np.array([i[3:125].astype('float32') for i in vector])

def anti_nan_vector(vector):
    where_are_NaNs = isnan(vector)
    vector[where_are_NaNs] = 0
    return vector

def anti_inf_vector(vector):
    where_are_infs = isinf(vector)
    vector[where_are_infs] = 0
    return vector

def feature_selection(vectors_labels, index, preferred, style='recursive'):
    if style not in ['tree', 'chi2', 'recursive']:
        return 'Error: style of feature selection not recognized.'
    # Splitting the vector_labels into actual vectors and labels
    vectors, labels = create_labeled_vectors(vectors_labels)
    # Floatify the vectors
    vectors = floatify_vector(vectors)
    vectors = anti_inf_vector(anti_nan_vector(vectors))
    # Do feature Selection
    if style == 'recurseive':
        fitting = recursive_selection(vectors, labels, preferred)
        fitting = np.array(fitting)
        return index[fitting]
    elif style == 'tree':
        fitting = extra_trees(vectors, labels)
        test = [(i,j) for i,j in enumerate(fitting)]
        test = sorted(test, key=lambda x: x[1])
        indices = [i[0] for i in test]
        return np.array([index[i] for i in indices])
    else:
        fitting = chi2_selection(vectors, labels, preferred)
        return fitting


# ----------------------------------------------------------------
# VISUALIZATION FOR NORMALITY / FEATURES  / ERROR

import statsmodels.api as sm
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.lda import LDA

def probplot(data):
    stats.probplot(data, dist="norm", plot=pylab)
    pylab.show()

def qqplot(data):
    sm.qqplot(data)
    pylab.show()

# Pearson requires two normally distributed datasets
# Kendall doesn't require too much
# Nonparametric form

# Instead of worrying about what's what. I'm just going to try all of these three
def get_feature_correlation(feature1, feature2, style="pearson", tails="one"):
    if style not in ["pearson", "kendall", "spearman"]:
        return 'Error: The style provided is not allowed.'
    if tails not in ["one", "two"]:
        return 'Error: The tail provided is not allowed.'
    if style == "pearson":
        stat, prob = pearsonr(feature1, feature2)
    elif style == "kendall":
        stat, prob = kendalltau(feature1, feature2)
    else:
        stat, prob = spearmanr(feature1, feature2)
    
    if tails == "two":
        return stat, prob
    else:
        return stat, prob/2

# PCA = generate component axes that maximize the variance
# LDA = Maximizing the component axes for class separation
# sklearn_pca = sklearnPCA(n_components=2)
# X_pca = sklearn_pca.fit_transform(X)
def plot_pca():
    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_pca[:,0][y == label],
                y=X_pca[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label])

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('PCA: Iris projection onto the first 2 principal components')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 
    ax.spines["left"].set_visible(False)

    plt.tight_layout
    plt.grid()
    plt.show()

# DEPRECATED ... NOT IN USE CURRENTLY
def plot_step_lda():
    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:,0][y == label],
                y=X_lda[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()

def plot_scikit_lda(X, title, mirror=1):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X[:,0][y == label]*mirror,
                y=X[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.spines["bottom"].set_visible(False) 
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()

def combine_lda_pca(X, y):
    sklearn_lda = LDA(n_components=2)
    X_lda_sklearn = sklearn_lda.fit_transform(X, y)
    sklearn_pca = sklearnPCA(n_components=2) #PCA
    X_ldapca_sklearn = sklearn_pca.fit_transform(X_lda_sklearn)
    plot_scikit_lda(X_ldapca_sklearn, title='LDA+PCA via scikit-learn', mirror=(-1))

# ----------------------------------------------------------------
# Self organizing map code 

def SOM_clustering(pruned, xsize, ysize, plotting=True):
    msz0 = xsize
    msz1 = ysize
    # Hack to get rid of these
    pruned[isnan(pruned)] = 0
    pruned[isinf(pruned)] = 0
    sm = SOMPY.SOM('sm', pruned, mapsize=[msz0, msz1], norm_method = 'var')
    sm.train(n_job = 1, shared_memory = 'no')
    if plotting == True:
        sm.view_map(which_dim = 'all')

# Splitting for Testing and Training
def split_test_train(vector, label, size=0.3, random_seed=42):
    v_train, v_test, l_train, l_test = train_test_split(vector, label, test_size=0.3, random_state=42)
    return v_train, v_test, l_train, l_test