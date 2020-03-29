## unit tests for bmi_delta.py

import pytest

import os
import numpy as np
from pyDeltaRCM import BmiDelta

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml
delta = BmiDelta()

def test_initialize():
    '''
    test function BmiDelta.initialize
    '''
    delta.initialize(os.getcwd() + '/tests/test.yaml')
    assert delta._delta.f_bedload == 0.5
