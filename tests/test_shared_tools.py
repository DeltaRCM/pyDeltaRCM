# unit tests for shared_tools.py

import pytest

import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM
from pyDeltaRCM import shared_tools

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml to create a small test case
delta = pyDeltaRCM(input_file = os.path.join(os.getcwd(), 'tests', 'test.yaml'))

# now that it is initiated can access the shared_tools via the inherited object
# delta._delta.**shared_tools_function**


def test_random_pick():
    """
    Test for function shared_tools.random_pick
    """
    # define probs array of zeros with a single 1 value
    probs = np.zeros((8,))
    probs[0] = 1
    # should return first index
    assert delta.random_pick(probs) == 0


def test_random_pick_inlet():
    """
    Test for function shared_tools.random_pick_inlet
    """
    choices = [0]
    probs = np.ones((1,))
    assert delta.random_pick_inlet(choices,probs) == 0

