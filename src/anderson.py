import numpy as np
from scipy.stats import distributions


def anderson(x):
    y = np.sort(x)
    xbar = np.mean(x, axis=0)
    N = len(y)
    s = np.std(x, ddof=1, axis=0)
    w = (y - xbar) / s
    logcdf = distributions.norm.logcdf(w)
    logsf = distributions.norm.logsf(w)
    i = np.arange(1, N + 1)
    A2 = -N - np.sum((2*i - 1.0) / N * (logcdf + logsf[::-1]), axis=0)
    A2_star = np.around(A2 * (1.0 + 0.75 / N - 2.25 / (N ** 2)), 3)
    p = np.exp(1.2937 - 5.709 * A2_star + 0.0186 * A2_star ** 2)
    return p
