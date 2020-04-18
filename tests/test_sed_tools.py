# unit tests for sed_tools.py

import pytest

import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

from pyDeltaRCM import BmiDelta
from pyDeltaRCM import sed_tools

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml to create a small test case
delta = BmiDelta()
delta.initialize(os.getcwd() + '/tests/test.yaml')

# now that it is initiated can access the shared_tools via the inherited object
# delta._delta.**sed_tools_function**


def test_sed_route():
    '''
    test the function sed_tools.sed_route
    '''
    delta._delta.pad_cell_type = np.pad(delta._delta.cell_type, 1, 'edge')
    delta._delta.sed_route()
    [a, b] = np.shape(delta._delta.pad_depth)
    assert a == 12
