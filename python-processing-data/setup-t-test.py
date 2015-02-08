# This file contains the functions used to perform a 1 sample, 2 sample 
# parametric, and non-parametric t-test.

# Please see ipython notebook for visual context.

from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu

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
    return result