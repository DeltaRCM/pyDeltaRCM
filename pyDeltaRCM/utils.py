
import numpy as np

import time
import numba

# utilities used in various places in the model and docs


@numba.njit
def random_pick(probs):
    """
    Randomly pick a number weighted by array probs (len 8)
    Return the index of the selected weight in array probs
    """

    # catch bad prob arrays
    if np.nansum(probs) == 0:
        probs[1, 1] = 0
        for i, prob in np.ndenumerate(probs):
            if np.isnan(prob):
                probs[i] = 1

    # prep array
    for i, prob in np.ndenumerate(probs):
        if np.isnan(prob):
            probs[i] = 0

    cutoffs = np.cumsum(probs)
    v = np.random.uniform(0, cutoffs[-1])
    idx = np.searchsorted(cutoffs, v)

    return idx


@numba.njit
def random_pick_inlet(choices, probs=None):
    """
    Randomly pick a number from array choices weighted by array probs
    Values in choices are column indices

    Return a tuple of the randomly picked index for row 0
    """

    if probs is None:
        probs = np.array([1. for i in range(len(choices))])

    cutoffs = np.cumsum(probs)
    # print(cutoffs.dtype)
    v = np.random.uniform(0, cutoffs[-1])
    # print("TYPES:", cutoffs.dtype, type(v))
    idx = np.searchsorted(cutoffs, v)

    return choices[idx]


def _get_version():
    """
    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()
