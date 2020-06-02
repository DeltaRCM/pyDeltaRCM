# unit tests for water_tools.py

import pytest

import sys
import os
import numpy as np

from utilities import test_DeltaModel
from pyDeltaRCM import shared_tools


def test_update_flow_field_inlet(test_DeltaModel):
    """
    Check that the flow at the inlet is set as expected
    """
    test_DeltaModel.update_flow_field(1)
    assert test_DeltaModel.qw[0, 4] == 1.


def test_update_flow_field_out(test_DeltaModel):
    """
    Check that the flow in domain is set as expected when no flow (qx & qy==0)
    """
    test_DeltaModel.update_flow_field(1)
    assert test_DeltaModel.qw[0, 0] == 0.


def test_update_velocity_field(test_DeltaModel):
    """
    Check that flow velocity field is updated as expected
    """
    test_DeltaModel.update_velocity_field()
    assert test_DeltaModel.uw[0, 0] == 0.


def test_pad_stage(test_DeltaModel):
    """
    Test padding of stage field
    Padded shape will be initial length/width + 2 so [12,12]
    """
    test_DeltaModel.init_water_iteration()
    [a, b] = np.shape(test_DeltaModel.pad_stage)
    assert a == 12


def test_pad_depth(test_DeltaModel):
    """
    Test padding of depth field
    Padded shape will be initial length/width + 2 so [12,12]
    """
    test_DeltaModel.init_water_iteration()
    [a, b] = np.shape(test_DeltaModel.pad_depth)
    assert a == 12


def test_pad_cell_type(test_DeltaModel):
    """
    Test padding of cell_type field
    Padded shape will be initial length/width + 2 so [12,12]
    """
    test_DeltaModel.init_water_iteration()
    [a, b] = np.shape(test_DeltaModel.pad_cell_type)

    assert a == 12


def test_calculate_new_ind(test_DeltaModel):
    """
    Test for function water_tools.calculate_new_ind
    """
    # assign old index
    old_inds = np.array([4, 5])
    # assign new cell location
    new_cells = np.array([7, 7])
    # expect new cell to be in location (1,4) -> 14

    new_inds = shared_tools.calculate_new_ind(old_inds, new_cells,
                                              test_DeltaModel.iwalk.flatten(),
                                              test_DeltaModel.jwalk.flatten(),
                                              test_DeltaModel.eta.shape)
    assert np.all(new_inds == np.array([14, 15]))

    # assert test_DeltaModel.calculate_new_ind(old_ind, new_cell) == 14
