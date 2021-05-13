# unit tests for water_tools.py

import pytest

import numpy as np

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from . import utilities

from pyDeltaRCM import water_tools


class TestWaterRoute:

    def test_route_water(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock top-level methods
        _delta.log_info = mock.MagicMock()
        _delta.init_water_iteration = mock.MagicMock()
        _delta.run_water_iteration = mock.MagicMock()
        _delta.compute_free_surface = mock.MagicMock()
        _delta.finalize_water_iteration = mock.MagicMock()

        # run the method
        _delta.route_water()

        # methods called
        assert (_delta.log_info.called is True)
        assert (_delta.init_water_iteration.call_count == _delta._itermax)
        assert (_delta.run_water_iteration.call_count == _delta._itermax)
        assert (_delta.compute_free_surface.call_count == _delta._itermax)
        assert (_delta.finalize_water_iteration.call_count == _delta._itermax)


class TestWaterRoutingWeights:

    def test_get_weight_at_cell(self):
        # prepare necessary ingredients for test
        ind = (0, 4)
        np.random.seed(0)
        stage = np.random.uniform(0.5, 1, 9)
        eta = np.random.uniform(0, 0.85, 9)
        depth = stage - eta
        depth[depth < 0] = 0
        celltype = np.array([-2, -2, -2, 1, 1, -2, 0, 0, 0])
        dry_thresh = 0.1
        gamma = 0.001962
        theta = 1
        weight_sfc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        weight_int = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1], dtype=np.float64)

        # get weights back from the test
        wts = water_tools._get_weight_at_cell_water(
            ind, weight_sfc, weight_int, depth,
            celltype, dry_thresh, gamma, theta)
        assert np.all(wts[[0, 1, 2, 5]] == 0)
        assert wts[4] == 0
        assert np.any(wts[[3, 6, 7, 8]] != 0)


class TestInitWaterIteration:

    def test_fields_updated(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        delta.log_info = mock.MagicMock()

        # run the method
        delta.init_water_iteration()

        # assertions
        assert np.all(delta.qxn == 0)
        assert np.all(delta.qyn == 0)
        assert np.all(delta.qwn == 0)
        assert np.all(delta.free_surf_flag == 1)  # all parcels begin as valid
        assert np.all(delta.free_surf_walk_inds == 0)
        assert np.all(delta.sfc_visit == 0)
        assert np.all(delta.sfc_sum == 0)

        assert delta.log_info.call_count == 1


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


class TestCalculateNewInds:

    def test_calculate_new_inds(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        current_inds = np.array(
            [7754, 7743, 7755], dtype=np.int64)
        new_inds = np.array(
            [6, 1, 4], dtype=np.int64)
        ravel_walk_flat = delta.ravel_walk_flat

        nidx = water_tools._calculate_new_inds(
            current_inds, new_inds,
            ravel_walk_flat)

        nidx_exp = np.array([7754 + delta.eta.shape[1] - 1,
                             7743 + -delta.eta.shape[1],
                             0])
        assert np.all(nidx == nidx_exp)


class TestUpdateQFields:

    def test_update_dirQfield(self):
        # configure necessary ingredients
        np.random.seed(0)
        qx = np.random.uniform(0, 10, 9)
        d = np.array([1, np.sqrt(2), 0])
        astep = np.array([True, True, False])
        inds = np.array([3, 4, 5])
        stepdir = np.array([1, 1, 0])

        # run the method
        qxn = water_tools._update_dirQfield(
            np.copy(qx), d, inds, astep, stepdir)

        # compute the expected value
        qxdiff = qxn - qx
        qxdiff_exp = np.array([1, np.sqrt(2) / 2, 0])

        # assertions
        assert np.all(qxdiff[3:6] == pytest.approx(qxdiff_exp))

    def test_update_absQfield(self):
        # configure necessary ingredients
        np.random.seed(0)
        qw = np.random.uniform(0, 10, 9)
        d = np.array([1, np.sqrt(2), 0])
        astep = np.array([True, True, False])
        inds = np.array([3, 4, 5])
        Qp_water = 0.3
        dx = 1.0

        # run the method
        qwn = water_tools._update_absQfield(
            np.copy(qw), d, inds, astep,
            Qp_water, dx)

        # compute the expected values
        qwdiff = qwn - qw
        diffelem = Qp_water / dx / 2
        qwdiff_exp = np.array([diffelem, diffelem, 0])

        # assertions
        assert np.all(qwdiff[3:6] == pytest.approx(qwdiff_exp))


class TestUpdateFlowField:

    def test_update_flow_field_time0_iteration0(self, tmp_path):
        """
        Check that the flow at the inlet is set as expected
        """
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        # mock the log
        delta.log_info = mock.MagicMock()

        # conditions are zero already, but...
        delta._time_iter = 0
        iteration = 0

        # run the method
        delta.update_flow_field(iteration)

        # assertions
        assert np.all(delta.qx[1:, :] == delta.qxn[1:, :])
        assert np.all(delta.qy[1:, :] == delta.qyn[1:, :])

        # check inlet boundary conditon
        assert np.all(delta.qx[0, delta.inlet] == delta.qw0)
        assert np.all(delta.qy[0, delta.inlet] == 0)
        assert np.all(delta.qw[0, delta.inlet] == delta.qw0)

        assert delta.log_info.call_count == 1

    def test_update_flow_field_time1_iteration0(self, tmp_path):
        """
        Check that the flow in domain is set as expected when no flow (qx &
        qy==0)
        """
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        delta.log_info = mock.MagicMock()

        # conditions are zero already, but...
        delta._time_iter = 1
        iteration = 0

        # run the method
        qx0 = np.copy(delta.qx)
        delta.update_flow_field(iteration)

        # assertions
        #  not sure what to check here other than that something changed
        assert np.any(delta.qx != qx0)

        # check inlet boundary conditon
        assert np.all(delta.qx[0, delta.inlet] == delta.qw0)
        assert np.all(delta.qy[0, delta.inlet] == 0)
        assert np.all(delta.qw[0, delta.inlet] == delta.qw0)

        assert delta.log_info.call_count == 1

    def test_update_flow_field_time1_iteration1(self, tmp_path):
        """
        Check that the flow in domain is set as expected when no flow (qx &
        qy==0)
        """
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        delta.log_info = mock.MagicMock()

        # conditions are zero already, but...
        delta._time_iter = 1
        iteration = 1

        # run the method
        qx0 = np.copy(delta.qx)
        delta.update_flow_field(iteration)

        # assertions
        #  not sure what to check here other than that something changed
        assert np.any(delta.qx != qx0)

        # check inlet boundary conditon
        assert np.all(delta.qx[0, delta.inlet] == delta.qw0)
        assert np.all(delta.qy[0, delta.inlet] == 0)
        assert np.all(delta.qw[0, delta.inlet] == delta.qw0)

        assert delta.log_info.call_count == 1

    def test_update_velocity_field(self, tmp_path):
        """
        Check that flow velocity field is updated as expected
        """
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        delta.log_info = mock.MagicMock()

        # run the method
        delta.update_velocity_field()

        # make a mask, which is less restictive that the actual mask
        dmask = delta.depth > 0

        # assertions
        assert np.all(delta.ux[dmask] != 0)
        assert np.all(delta.uw[dmask] != 0)

        assert delta.log_info.call_count == 1
