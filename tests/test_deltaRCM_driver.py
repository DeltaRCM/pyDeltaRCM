## unit tests for deltaRCM_driver.py

import pytest

import sys, os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml
delta = pyDeltaRCM(input_file = os.path.join(os.getcwd(), 'tests', 'test.yaml'))

def test_init():
    '''
    test the deltaRCM_driver init (happened when delta.initialize was run)
    '''
    assert delta._time == 0.

def test_update():
    delta.update()
    assert delta._time == 1.0
