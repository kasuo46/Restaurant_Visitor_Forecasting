import numpy as np


def weighted_mean(mean1, ct1, mean2, ct2):
    if (ct1 == 0 or np.isnan(ct1)) and (ct2 == 0 or np.isnan(ct2)):
        return np.nan
    elif ct1 == 0 or np.isnan(ct1):
        return mean2
    elif ct2 == 0 or np.isnan(ct2):
        return mean1
    else:
        return (mean1 * ct1 + mean2 * ct2)/(ct1 + ct2)
