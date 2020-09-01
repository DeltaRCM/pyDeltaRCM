# unit tests for water_tools.py

import pytest

import sys
import os
import numpy as np

from utilities import test_DeltaModel
from pyDeltaRCM import water_tools
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

    new_inds = water_tools.calculate_new_ind(old_inds, new_cells,
                                             test_DeltaModel.iwalk.flatten(),
                                             test_DeltaModel.jwalk.flatten(),
                                             test_DeltaModel.eta.shape)
    assert np.all(new_inds == np.array([14, 15]))


def test_check_for_loops():

    idxs = np.array(
        [[0, 11, 12, 13, 23, 22, 12],
         [0, 1, 2, 3, 4, 5, 16]])
    nidx = np.array([21, 6])
    itt = 6
    free = np.array([1, 1])
    CTR = 4
    L0 = 1
    looped = np.array([0, 0])

    nidx, looped, free = water_tools._check_for_loops(
        idxs, nidx, itt, L0, looped, (10, 10), CTR, free)

    assert np.all(nidx == [41, 6])
    assert np.all(looped == [1, 0])
    assert np.all(free == [-1, 1])


def test_calculate_new_ind():

    cidx = np.array([12, 16, 16])
    ncel = np.array([6, 1, 4])
    iwalk = shared_tools.get_iwalk()
    jwalk = shared_tools.get_jwalk()

    nidx = water_tools._calculate_new_ind(cidx, ncel, iwalk.flatten(), jwalk.flatten(), (10, 10))

    nidx_exp = np.array([21, 6, 0])
    assert np.all(nidx == nidx_exp)


def test_update_dirQfield(test_DeltaModel):
    """
    Test for function water_tools._update_dirQfield
    """
    np.random.seed(test_DeltaModel.seed)
    qx = np.random.uniform(0, 10, 9)
    d = np.array([1, np.sqrt(2), 0])
    astep = np.array([True, True, False])
    inds = np.array([3, 4, 5])
    stepdir = np.array([1, 1, 0])
    qxn = water_tools._update_dirQfield(np.copy(qx), d, inds, astep, stepdir)
    qxdiff = qxn - qx
    qxdiff_exp = np.array([1, np.sqrt(2) / 2, 0])
    assert np.all(qxdiff[3:6] == pytest.approx(qxdiff_exp))


def test_update_absQfield(test_DeltaModel):
    """
    Test for function water_tools._update_absQfield
    """
    np.random.seed(test_DeltaModel.seed)
    qw = np.random.uniform(0, 10, 9)
    d = np.array([1, np.sqrt(2), 0])
    astep = np.array([True, True, False])
    inds = np.array([3, 4, 5])
    qwn = water_tools._update_absQfield(
        np.copy(qw), d, inds, astep, test_DeltaModel.Qp_water, test_DeltaModel.dx)
    qwdiff = qwn - qw
    diffelem = test_DeltaModel.Qp_water / test_DeltaModel.dx / 2
    qwdiff_exp = np.array([diffelem, diffelem, 0])
    assert np.all(qwdiff[3:6] == pytest.approx(qwdiff_exp))