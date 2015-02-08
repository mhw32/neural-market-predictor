# This file contains the functions used to perform a 1 sample, 2 sample 
# parametric, and non-parametric t-test.

# Please see ipython notebook for visual context.

from scipy.stats import ttest_1samp

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
        