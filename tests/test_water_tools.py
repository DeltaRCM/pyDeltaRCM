# unit tests for water_tools.py

import pytest

import sys
import os
import numpy as np

from utilities import test_DeltaModel
from pyDeltaRCM import water_tools
from pyDeltaRCM import shared_tools


class TestWaterRoutingWeights:

    def test_get_weight_at_cell(self, test_DeltaModel):
        ind = (0, 4)
        np.random.seed(test_DeltaModel.seed)
        stage = np.random.uniform(0.5, 1, 9)
        eta = np.random.uniform(0, 0.85, 9)
        depth = stage - eta
        depth[depth < 0] = 0
        celltype = np.array([-2, -2, -2, 1, 1, -2, 0, 0, 0])
        qx = 1
        qy = 1
        ivec = test_DeltaModel.ivec.flatten()
        jvec = test_DeltaModel.jvec.flatten()
        dists = test_DeltaModel.distances.flatten()
        dry_thresh = 0.1
        gamma = 0.001962
        theta = 1
        weight_sfc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        weight_int = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1], dtype=np.float64)

        wts = water_tools._get_weight_at_cell_water(ind, weight_sfc, weight_int, depth,
                                                    celltype, dry_thresh, gamma, theta)
        assert np.all(wts[[0, 1, 2, 5]] == 0)
        assert wts[4] == 0
        assert np.any(wts[[3, 6, 7, 8]] != 0)


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


class TestCheckForLoops:

    # set up a simple domain
    #   10x10 domain
    #   2 parcels taken six steps
    free_surf_walk_inds = np.array(
            [[4, 15, 25, 36, 46, 55, 66],
             [5, 15, 26, 35, 44, 53, 62]]
             )
    stage = np.ones((10, 10))
    CTR = 4
    L0 = 1

    def test_check_for_loops_no_loops(self):

        # set up inds to go to new locations
        new_inds0 = np.array([76, 73])
        _step = self.free_surf_walk_inds.shape[1] + 1  # 7
        SL = 0
        stage_minus_SL = self.stage - SL

        new_inds, looped = water_tools._check_for_loops(
            self.free_surf_walk_inds, new_inds0.copy(), _step,
            self.L0, self.CTR, stage_minus_SL)

        # new inds should be same as input
        assert np.all(new_inds == new_inds0)
        # no loops
        assert np.all(looped == [0, 0])

    def test_check_for_loops_relocate_not_loop(self):
        """
        Will relocate parcel (repeated ind), but will not mark as a loop,
        because parcel stage is at SL.
        """
        # set up inds[1] to go to a repeat ind
        new_inds0 = np.array([76, 53])
        _step = self.free_surf_walk_inds.shape[1] + 1  # 7
        SL = 1
        stage_minus_SL = self.stage - SL

        new_inds, looped = water_tools._check_for_loops(
            self.free_surf_walk_inds, new_inds0.copy(), _step,
            self.L0, self.CTR, stage_minus_SL)

        # new inds should NOT be same as input
        assert new_inds[0] == new_inds0[0]
        assert new_inds[1] != new_inds0[1]
        # check that the parcel was thrown along vector
        assert new_inds[1] > 80
        # one loop
        assert np.all(looped == [0, 0])

    def test_check_for_loops_relocate_and_loop(self):
        """
        Will relocate parcel (repeated ind), and mark as a loop, because
        parcel stage is not at SL.
        """
        # set up inds[1] to go to a repeat ind
        new_inds0 = np.array([76, 53])
        _step = self.free_surf_walk_inds.shape[1] + 1  # 7
        SL = 0
        stage_minus_SL = self.stage - SL

        new_inds, looped = water_tools._check_for_loops(
            self.free_surf_walk_inds, new_inds0.copy(), _step,
            self.L0, self.CTR, stage_minus_SL)

        # new inds should NOT be same as input
        assert new_inds[0] == new_inds0[0]
        assert new_inds[1] != new_inds0[1]
        # check that the parcel was thrown along vector
        assert new_inds[1] > 80
        # one loop
        assert np.all(looped == [0, 1])


def test_calculate_new_ind(test_DeltaModel):

    current_inds = np.array([12, 16, 16], dtype=np.int64)
    new_inds = np.array([6, 1, 4], dtype=np.int64)
    ravel_walk_flat = test_DeltaModel.ravel_walk_flat

    nidx = water_tools._calculate_new_inds(
        current_inds, new_inds,
        ravel_walk_flat
        )

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