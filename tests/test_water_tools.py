## unit tests for test_water_tools.py

import pytest

import os
import numpy as np
from pyDeltaRCM import BmiDelta
from pyDeltaRCM import water_tools

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml to create a small test case
delta = BmiDelta()
delta.initialize(os.getcwd() + '/tests/test.yaml')

# now that it is initiated can access the water_tools via the inherited object
# delta._delta.**water_tools_function**

def test_update_flow_field_inlet():
    '''
    Check that the flow at the inlet is set as expected
    '''
    delta._delta.update_flow_field(1)
    assert delta._delta.qw[0,4] == 1.
