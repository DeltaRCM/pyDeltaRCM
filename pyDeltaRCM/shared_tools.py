##  tools shared between deltaRCM water and sediment routing
##  refactored by eab
##  Jan 2020

from math import floor, sqrt, pi
import numpy as np

import time

class shared_tools(object):

    def random_pick(self, probs):
        '''
        Randomly pick a number weighted by array `probs` (len 8)

        Parameters
        ----------

        probs : `ndarray`
            An 8, `ndarray` with the weighted probabilities associated with each of the 8 neighboring cells

        Returns
        -------

        idx : `int`
            The index of the selected weight from the array `probs`
        '''

        num_nans = sum(np.isnan(probs))

        if np.nansum(probs) == 0:
            probs[~np.isnan(probs)] = 1
            probs[1,1] = 0

        probs[np.isnan(probs)] = 0
        cutoffs = np.cumsum(probs)
        idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))

        return idx

    def random_pick_inlet(self, choices, probs = None):
        '''
        Randomly pick a number from array choices weighted by array `probs` as the inlet location for a given parcel

        Parameters
        ----------

        choices : `list`
            Values in choices are column indices corresponding to inlet cell locations

        probs : `ndarray`, optional
            An 8, `ndarray` with the weighted probabilities associated with each of the 8 neighboring cells; default is an array of 1s to weight each location equally.

        Returns
        -------

        choices[idx] : `int`
            The randomly picked index for row 0 that the parcel will start at 
        '''

        if not probs:
            probs = np.array([1 for i in range(len(choices))])

        cutoffs = np.cumsum(probs)
        idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))

        return choices[idx]
