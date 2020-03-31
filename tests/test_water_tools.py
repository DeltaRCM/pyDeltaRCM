## unit tests for water_tools.py

import pytest

import sys, os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

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

def test_update_flow_field_out():
    '''
    Check that the flow in domain is set as expected when no flow (qx & qy==0)
    '''
    delta._delta.update_flow_field(1)
    assert delta._delta.qw[0,0] == 0.

def test_update_velocity_field():
    '''
    Check that flow velocity field is updated as expected
    '''
    delta._delta.update_velocity_field()
    assert delta._delta.uw[0,0] == 0.

def test_pad_stage():
    '''
    Test padding of stage field
    Padded shape will be initial length/width + 2 so [12,12]
    '''
    delta._delta.init_water_iteration()
    [a,b] = np.shape(delta._delta.pad_stage)
    assert a == 12

def test_pad_depth():
    '''
    Test padding of depth field
    Padded shape will be initial length/width + 2 so [12,12]
    '''
    delta._delta.init_water_iteration()
    [a,b] = np.shape(delta._delta.pad_depth)
    assert a == 12

def test_pad_cell_type():
    '''
    Test padding of cell_type field
    Padded shape will be initial length/width + 2 so [12,12]
    '''
    delta._delta.init_water_iteration()
    [a,b] = np.shape(delta._delta.pad_cell_type)
    assert a == 12

def test_calculate_new_ind():
    '''
    Test for function water_tools.calculate_new_ind
    '''
    # assign old index
    old_ind = [0,4]
    # assign new cell location
    new_cell = 7
    # expect new cell to be in location (1,4) -> 14
    assert delta._delta.calculate_new_ind(old_ind,new_cell) == 14
