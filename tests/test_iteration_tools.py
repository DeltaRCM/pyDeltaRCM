# unit tests for deltaRCM_tools.py

import pytest

import glob
import os
import netCDF4
import numpy as np

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
import utilities


class TestRunOneTimestep:

    def test_run_one_timestep_defaults(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        # mock top-level methods, verify call was made to each
        delta.init_water_iteration = mock.MagicMock()
        delta.run_water_iteration = mock.MagicMock()
        delta.compute_free_surface = mock.MagicMock()
        delta.finalize_water_iteration = mock.MagicMock()
        delta.sed_route = mock.MagicMock()

        # run the timestep
        delta.run_one_timestep()

        # assert that methods are called
        assert delta.init_water_iteration.called is True
        assert delta.run_water_iteration.called is True
        assert delta.compute_free_surface.called is True
        assert delta.finalize_water_iteration.called is True
        _calls = [mock.call(0), mock.call(1), mock.call(2)]
        delta.finalize_water_iteration.assert_has_calls(
            _calls, any_order=False)
        assert delta.finalize_water_iteration.call_count == 3
        assert (delta.sed_route.called is True)
        assert (delta._is_finalized is False)

    def test_run_one_timestep_itermax_10(self, tmp_path):
        # create a delta with different itermax
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'itermax': 10})
        delta = DeltaModel(input_file=p)

        # mock top-level methods, verify call was made to each
        delta.init_water_iteration = mock.MagicMock()
        delta.run_water_iteration = mock.MagicMock()
        delta.compute_free_surface = mock.MagicMock()
        delta.finalize_water_iteration = mock.MagicMock()
        delta.sed_route = mock.MagicMock()

        # run the timestep
        delta.run_one_timestep()

        # assert that methods are called
        assert delta.init_water_iteration.called is True
        assert delta.run_water_iteration.called is True
        assert delta.compute_free_surface.called is True
        assert delta.finalize_water_iteration.called is True
        _calls = [mock.call(i) for i in range(10)]
        delta.finalize_water_iteration.assert_has_calls(
            _calls, any_order=False)
        assert delta.finalize_water_iteration.call_count == 10
        assert (delta.sed_route.called is True)
        assert (delta._is_finalized is False)


class TestFinalizeTimestep:

    def test_finalize_timestep(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'SLR': 0.001})
        delta = DeltaModel(input_file=p)
        # mock the flooding correction
        delta.flooding_correction = mock.MagicMock()
        # run the step
        delta.finalize_timestep()
        # assert submethod called once
        delta.flooding_correction.call_count == 1
        # check that sea level rose as expected
        assert delta.H_SL == 25


class TestApplyingSubsidence:

    in_idx = (34, 44)
    out_idx = (34, 45)

    def test_subsidence_in_update(self, tmp_path):
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'toggle_subsidence': True,
                                      'sigma_max': 1e-8,
                                      'start_subsidence': 0,
                                      'seed': 0})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        assert _delta.dt == 25000
        assert _delta.sigma_max == 1e-8
        assert _delta.sigma[self.in_idx] == 0.0  # outside the sigma mask
        assert _delta.sigma[self.out_idx] == 0.00025  # inside the sigma mask
        assert np.all(_delta.eta[34, 44:7] == -_delta.h0)

        _delta.update()
        assert _delta.eta[self.in_idx] == pytest.approx(-_delta.h0)
        assert _delta.eta[self.out_idx] == pytest.approx(-_delta.h0 - 0.00025)
        _delta.output_netcdf.close()

    def test_subsidence_in_update_delayed_start(self, tmp_path):
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'toggle_subsidence': True,
                                      'sigma_max': 1e-8,
                                      'start_subsidence': 25000,
                                      'seed': 0})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        assert _delta.dt == 25000
        assert _delta.sigma_max == 1e-8
        assert _delta.sigma[self.in_idx] == 0.0  # outside the sigma mask
        assert _delta.sigma[self.out_idx] == 0.00025  # inside the sigma mask
        assert np.all(_delta.eta[self.in_idx[0],
                                 self.in_idx[1]:self.in_idx[1]+5] ==
                      -_delta.h0)

        _delta.update()  # no subsidence applied
        assert _delta.time == 25000
        assert _delta.eta[self.in_idx] == pytest.approx(-_delta.h0)
        assert _delta.eta[self.out_idx] == pytest.approx(-_delta.h0)

        _delta.update()
        assert _delta.time == 50000
        assert _delta.eta[self.in_idx] == pytest.approx(-_delta.h0)
        assert _delta.eta[self.out_idx] == pytest.approx(-_delta.h0 - 0.00025)
        _delta.output_netcdf.close()

        _delta.run_one_timestep.call_count == 2

    def test_subsidence_changed_with_timestep(self, tmp_path):
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'toggle_subsidence': True,
                                      'sigma_max': 1e-8})
        _delta = DeltaModel(input_file=p)

        assert _delta.dt == 25000
        assert _delta.sigma[self.out_idx] == 0.00025
        # use the model setter to adjust the timestep
        _delta.time_step = 86400
        assert _delta.sigma[self.out_idx] == 0.000864
        _delta.output_netcdf.close()


class TestExpandingStratigraphy:

    def test_expand_stratigraphy(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'toggle_subsidence': True,
                                      'start_subsidence': 20000,
                                      'sigma_max': 1e-8,
                                      'save_dt': 50000})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        assert _delta.dt == 25000
        assert _delta.n_steps == 10
        assert _delta.strata_counter == 0
        assert _delta.strata_eta[:, _delta.strata_counter].getnnz() == 0
        for _t in range(19):
            assert _delta.strata_eta[:, _delta.strata_counter].getnnz() == 0
            _delta.update()
            assert _delta.time == _delta.dt * (_t + 1)
            assert _delta.strata_eta.shape[1] == 10
        assert _delta.time == 19 * 25000
        assert _delta.strata_counter == 10  # stored 10, invalid next store
        assert _delta.strata_eta.shape[1] == 10
        # nothing occurs on next  update, because save_dt = 2 * dt
        _delta.update()
        assert _delta.time == 20 * 25000
        assert _delta.strata_counter == 10
        assert _delta.strata_eta.shape[1] == 10
        # expansion occurs when model tries to save strata after next update
        _delta.update()
        assert _delta.time == 21 * 25000
        assert _delta.strata_counter == 11
        assert _delta.strata_eta.shape[1] == 20
        # run to bring to even 100 steps, check status again
        for _t in range(79):
            _delta.update()
        assert _delta.time == 100 * 25000
        assert _delta.strata_counter == 50
        assert _delta.strata_eta.shape[1] == 50


class TestOutputData:

    def test_save_no_figs_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        # check nothing created at the start
        img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
        nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
        assert len(img_glob) == 0
        assert len(nc_glob) == 0

        # update the delta a few times
        for _t in range(0, 4):
            _delta.update()

        # check nothing after a number of iterations, greater than dt
        assert _delta._time > _delta.save_dt
        assert _delta.time_iter == 4.0
        _delta.run_one_timestep.call_count == 4
        img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
        nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
        assert len(img_glob) == 0
        assert len(nc_glob) == 0

    def test_save_one_fig_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False,
                                      'save_eta_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        # check nothing created at the start
        img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
        nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
        assert len(img_glob) == 0
        assert len(nc_glob) == 0

        # update the delta a few times
        for _ in range(0, 2):
            _delta.update()

        assert _delta.time_iter == 2.0
        _delta.run_one_timestep.call_count == 2

        # check for output eta files
        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'out_dir', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'out_dir', 'eta_00001.png')
        exp_path_png2 = os.path.join(tmp_path / 'out_dir', 'eta_00002.png')
        assert not os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)
        assert not os.path.isfile(exp_path_png2)

    def test_save_one_fig_one_grid(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_eta_grids': True,
                                      'save_discharge_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)
        nc_size_before = os.path.getsize(exp_path_nc)
        assert nc_size_before > 0

        # update a couple times, should increase on each save
        for _ in range(0, 2):
            _delta.update()
        nc_size_middle = os.path.getsize(exp_path_nc)
        assert _delta.time_iter == 2.0
        assert nc_size_middle > nc_size_before

        # now finalize, and file size should increase over middle again
        _delta.finalize()
        nc_size_after = os.path.getsize(exp_path_nc)
        assert _delta.time_iter == 2.0
        assert nc_size_after > nc_size_middle
        assert nc_size_after > nc_size_before

    def test_save_all_figures_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False,
                                      'save_eta_figs': True,
                                      'save_discharge_figs': True,
                                      'save_velocity_figs': True,
                                      'save_stage_figs': True,
                                      'save_depth_figs': True,
                                      'save_sedflux_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert not os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()

        assert (_delta.run_one_timestep.call_count == 2)

        exp_path_png0 = os.path.join(
            tmp_path / 'out_dir', 'eta_00000.png')
        exp_path_png1 = os.path.join(
            tmp_path / 'out_dir', 'depth_00000.png')
        exp_path_png2 = os.path.join(
            tmp_path / 'out_dir', 'stage_00000.png')
        exp_path_png3 = os.path.join(
            tmp_path / 'out_dir', 'velocity_00000.png')
        exp_path_png4 = os.path.join(
            tmp_path / 'out_dir', 'discharge_00000.png')
        exp_path_png5 = os.path.join(
            tmp_path / 'out_dir', 'sedflux_00000.png')
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)
        assert os.path.isfile(exp_path_png2)
        assert os.path.isfile(exp_path_png3)
        assert os.path.isfile(exp_path_png4)
        assert os.path.isfile(exp_path_png5)

    def test_save_all_figures_sequential_false(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False,
                                      'save_eta_figs': True,
                                      'save_discharge_figs': True,
                                      'save_velocity_figs': True,
                                      'save_stage_figs': True,
                                      'save_depth_figs': True,
                                      'save_sedflux_figs': True,
                                      'save_figs_sequential': False})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert not os.path.isfile(exp_path_nc)

        for _ in range(0, 5):
            _delta.update()

        _delta.run_one_timestep.call_count == 5

        exp_path_png0 = os.path.join(
            tmp_path / 'out_dir', 'eta_00000.png')
        exp_path_png1 = os.path.join(
            tmp_path / 'out_dir', 'depth_00000.png')
        exp_path_png0_latest = os.path.join(
            tmp_path / 'out_dir', 'eta_latest.png')
        exp_path_png1_latest = os.path.join(
            tmp_path / 'out_dir', 'depth_latest.png')
        exp_path_png2_latest = os.path.join(
            tmp_path / 'out_dir', 'stage_latest.png')
        exp_path_png3_latest = os.path.join(
            tmp_path / 'out_dir', 'velocity_latest.png')
        exp_path_png4_latest = os.path.join(
            tmp_path / 'out_dir', 'discharge_latest.png')
        exp_path_png5_latest = os.path.join(
            tmp_path / 'out_dir', 'sedflux_latest.png')
        assert not os.path.isfile(exp_path_png0)
        assert not os.path.isfile(exp_path_png1)
        assert os.path.isfile(exp_path_png0_latest)
        assert os.path.isfile(exp_path_png1_latest)
        assert os.path.isfile(exp_path_png2_latest)
        assert os.path.isfile(exp_path_png3_latest)
        assert os.path.isfile(exp_path_png4_latest)
        assert os.path.isfile(exp_path_png5_latest)

    def test_save_metadata_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_metadata': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()

        assert _delta.time_iter == 2.0
        _delta.run_one_timestep.call_count == 2

        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        assert not ('eta' in ds.variables)
        assert ds['meta']['H_SL'].shape[0] == 3
        assert ds['meta']['L0'][:] == 3

    def test_save_metadata_and_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_metadata': True,
                                      'save_eta_grids': True,
                                      'save_velocity_grids': True,
                                      'f_bedload': 0.25})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()
        assert _delta.time_iter == 2.0
        _delta.run_one_timestep.call_count == 2

        _delta.finalize()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        assert ('eta' in ds.variables)
        assert ('velocity' in ds.variables)
        assert ds['meta']['H_SL'].shape[0] == 3
        assert ds['meta']['L0'][:] == 3
        assert np.all(ds['meta']['f_bedload'][:] == 0.25)

    def test_save_one_grid_metadata_by_default(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_metadata': False,
                                      'save_eta_grids': True,
                                      'C0_percent': 0.2})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 6):
            _delta.update()

        assert _delta.time_iter == 6.0
        _delta.run_one_timestep.call_count == 6

        _delta.finalize()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['eta']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too
        assert ds.groups['meta']['H_SL'].shape[0] == _arr.shape[0]
        assert np.all(ds.groups['meta']['C0_percent'][:] == 0.2)
        assert np.all(ds.groups['meta']['f_bedload'][:] == 0.5)


class TestValidityOfSavedGrids:

    def test_save_eta_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()
        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['eta']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too

    def test_save_depth_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_depth_grids': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()
        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['depth']
        assert _arr.shape[1] == _delta.depth.shape[0]
        assert _arr.shape[2] == _delta.depth.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too

    def test_save_velocity_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_velocity_grids': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()
        assert _delta.time_iter == 2.0
        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['velocity']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too

    def test_save_stage_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_stage_grids': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()
        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['stage']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too

    def test_save_discharge_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_discharge_grids': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()
        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['discharge']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too

    def test_save_sedflux_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_sedflux_grids': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()
        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['sedflux']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too
