
from math import floor, sqrt, pi
import numpy as np

import time

# tools shared between deltaRCM water and sediment routing


class shared_tools(object):

    def random_pick(self, probs):
        """
        Randomly pick a number weighted by array probabilities (len 9)
        Return the index of the selected weight in array probs
        Takes a numpy array that is the precalculated cumulative probability
        around the cell flattened to 1D.
        """

        idx = probs.searchsorted(np.random.uniform(0, probs[-1]))

        return idx

    def random_pick_inlet(self, choices, probs=None):
        """
        Randomly pick a number from array choices weighted by array probs
        Values in choices are column indices

        Return a tuple of the randomly picked index for row 0
        """

        if not probs:
            probs = np.ones(len(choices))

        cutoffs = np.cumsum(probs)
        idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))

        return choices[idx]


def _get_version():
    """
    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()
