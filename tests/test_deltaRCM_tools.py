## unit tests for deltaRCM_tools.py

import pytest

import os
import numpy as np
from pyDeltaRCM import BmiDelta
from pyDeltaRCM import Tools

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml to create a small test case
delta = BmiDelta()
delta.initialize(os.getcwd() + '/tests/test.yaml')

# now that it is initiated can access the shared_tools via the inherited object
# delta._delta.**Tools_function**

def test_get_var_name_depth():
    '''
    test get_var_name function from deltaRCM_tools
    '''
    assert delta._delta.get_var_name('channel__width') == 'N0_meters'
