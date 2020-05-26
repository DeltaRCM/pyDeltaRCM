# unit tests for shared_tools.py

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM
from pyDeltaRCM import shared_tools

# need to create a simple case of pydeltarcm object to test these functions
delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

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
    assert shared_tools.random_pick(probs) == 0
