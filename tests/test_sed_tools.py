## unit tests for sed_tools.py

import pytest

import sys, os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM
from pyDeltaRCM import sed_tools

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml to create a small test case
delta = pyDeltaRCM(input_file = os.path.join(os.getcwd(), 'tests', 'test_separated.yaml'))

# now that it is initiated can access the shared_tools via the inherited object
# delta._delta.**sed_tools_function**

def test_sed_route():
    '''
    test the function sed_tools.sed_route
    '''
    delta.pad_cell_type = np.pad(delta.cell_type, 1, 'edge')
    delta.sed_route()
    [a,b] = np.shape(delta.pad_depth)
    assert a == 12
