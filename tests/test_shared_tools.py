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
def test_custom_unravel_square():
    arr = np.arange(9).reshape((3, 3))
    # test upper left corner
    x, y = shared_tools.custom_unravel(0, arr.shape)
    assert x == 0
    assert y == 0
    # test center
    x, y = shared_tools.custom_unravel(4, arr.shape)
    assert x == 1
    assert y == 1
    # test off-center
    x, y = shared_tools.custom_unravel(5, arr.shape)
    assert x == 1
    assert y == 2
    # test lower right corner
    x, y = shared_tools.custom_unravel(8, arr.shape)
    assert x == 2
    assert y == 2


def test_custom_unravel_rectangle():
    arr = np.arange(50).reshape((5, 10))
    # test a few spots
    x, y = shared_tools.custom_unravel(19, arr.shape)
    assert x == 1
    assert y == 9
    x, y = shared_tools.custom_unravel(34, arr.shape)
    assert x == 3
    assert y == 4


@pytest.mark.xfail(raises=IndexError, strict=True)
def test_custom_unravel_exceed_error():
    arr = np.arange(9).reshape((3, 3))
    x, y = shared_tools.custom_unravel(99, arr.shape)
