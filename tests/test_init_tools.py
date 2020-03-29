## unit tests for init_tools.py

import pytest

import os
import numpy as np
from pyDeltaRCM import BmiDelta
from pyDeltaRCM import init_tools

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml to create a small test case
delta = BmiDelta()
delta.initialize(os.getcwd() + '/tests/test.yaml')

# now that it is initiated can access the init_tools via the inherited object
# delta._delta.**init_tools_function**

def test_set_constant_g():
    '''
    check gravity
    '''
    delta._delta.set_constants()
    assert delta._delta.g == 9.81
