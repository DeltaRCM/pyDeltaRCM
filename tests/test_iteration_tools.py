# unit tests for deltaRCM_tools.py

import pytest

import glob
import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from . import utilities


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


class TestOutputCheckpoint:

    def test_save_a_checkpoint_checkpoint_false(self, tmp_path):
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': False})
        _delta = DeltaModel(input_file=p)

        # mock the actual save checkpoint function to see if it was called
        _delta.save_the_checkpoint = mock.MagicMock()

        _delta.output_checkpoint()

        # assertions
        assert (_delta.save_the_checkpoint.called is False)
        assert (_delta._save_time_since_checkpoint == 0)

    def test_save_a_checkpoint_checkpoint_true(self, tmp_path):
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True})
        _delta = DeltaModel(input_file=p)

        # mock the actual save checkpoint function to see if it was called
        _delta.save_the_checkpoint = mock.MagicMock()

        # run the output checkpoint func
        _delta.output_checkpoint()

        # assertions
        assert (_delta.save_the_checkpoint.call_count == 1)
        assert (_delta._save_time_since_checkpoint == 0)

    def test_save_a_checkpoint_checkpoint_true_timewarning(self, tmp_path):
        # create a delta with subsidence parameters
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'checkpoint_dt': 864000000})
        _delta = DeltaModel(input_file=p)

        # mock the actual save checkpoint function to see if it was called
        _delta.save_the_checkpoint = mock.MagicMock()

        # this warning is only written to log, so mock the logger
        _delta.logger = mock.MagicMock()

        # run the output checkpoint func
        _delta.output_checkpoint()

        # assertions
        assert (_delta.save_the_checkpoint.call_count == 1)
        assert (_delta.logger.warning.call_count == 1)
        assert (_delta._save_time_since_checkpoint == 0)


class TestExpandingStratigraphy:

    def test_expand_stratigraphy(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # check initials
        assert _delta.dt == 25000
        assert _delta.n_steps == 15
        assert _delta.strata_counter == 0
        assert _delta.strata_eta.getnnz() == 0

        # check size before
        assert _delta.strata_eta.shape[1] == _delta.n_steps
        assert _delta.strata_sand_frac.shape[1] == _delta.n_steps

        # run the method
        _delta.expand_stratigraphy()

        # check that the size of the arry has increased
        assert _delta.strata_eta.shape[1] == _delta.n_steps * 2
        assert _delta.strata_sand_frac.shape[1] == _delta.n_steps * 2

        assert (_delta.log_info.call_count == 1)


class TestRecordStratigraphy:

    def test_record_stratigraphy_false(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_strata': False})
        _delta = DeltaModel(input_file=p)

        # mock the log_info and expanding
        _delta.log_info = mock.MagicMock()
        _delta.expand_stratigraphy = mock.MagicMock()

        assert not hasattr(_delta, 'strata_counter')
        assert not hasattr(_delta, 'strata_eta')

        # should do nothing, because save_strata False
        _delta.record_stratigraphy()

        # assertions
        assert (_delta.log_info.called is False)
        assert (_delta.expand_stratigraphy.called is False)

    def test_record_stratigraphy(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock the log_info and expanding
        _delta.log_info = mock.MagicMock()
        _delta.expand_stratigraphy = mock.MagicMock()

        assert _delta.dt == 25000
        assert _delta.n_steps == 15
        assert _delta.strata_counter == 0
        assert _delta.strata_sand_frac.getnnz() == 0

        # update the sand frac field so something is filled
        _rand = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.Vp_dep_sand += _rand

        # should record
        _delta.record_stratigraphy()

        assert _delta.strata_counter == 1
        assert _delta.strata_sand_frac.getnnz() > 0  # filled

        assert _delta.log_info.call_count == 1
        assert _delta.expand_stratigraphy.call_count == 0

    def test_record_stratigraphy_calls_expand(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock expanding, and error on expanding.
        #   just want to see that method is called, the actual method is
        #   tested elsewhere
        _delta.expand_stratigraphy = mock.MagicMock(
            side_effect=NotImplementedError)

        assert _delta.n_steps == 15
        assert _delta.strata_counter == 0
        assert _delta.strata_eta.shape[1] == 15

        # change counter to trigger expanding
        _delta.strata_counter = (_delta.n_steps)

        # should record
        with pytest.raises(NotImplementedError):
            _delta.record_stratigraphy()

        # should call expanding
        assert _delta.expand_stratigraphy.call_count == 1


class TestOutputData:

    def test_save_no_figs_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False})
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        # check nothing created at the start
        _delta.make_figure.call_count == 0
        _delta.save_figure.call_count == 0
        _delta.save_grids.call_count == 0

        # update the delta a few times
        for _t in range(0, 4):
            _delta._time = (_t * _delta._dt)
            _delta.output_data()

        # check nothing after a number of iterations, greater than dt
        assert _delta._time > _delta.save_dt
        _delta.make_figure.call_count == 0
        _delta.save_figure.call_count == 0
        _delta.save_grids.call_count == 0

    def test_save_one_fig_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False,
                                      'save_eta_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        # check nothing created at the start
        _delta.make_figure.call_count == 0
        _delta.save_figure.call_count == 0
        _delta.save_grids.call_count == 0

        assert (_delta._save_any_figs is True)
        assert (_delta._save_eta_figs is True)

        # update the delta a few times
        for _t in range(0, 5):
            _delta.output_data()
            _delta._save_iter += 1

        _delta.make_figure.call_count == 5
        _delta.save_figure.call_count == 5
        _delta.save_grids.call_count == 0

    def test_save_one_fig_one_grid(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_eta_grids': True,
                                      'save_discharge_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        # check for the netcdf file
        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)
        nc_size_before = os.path.getsize(exp_path_nc)
        assert nc_size_before > 0

        # update a couple times, should increase on each save
        for _t in range(0, 5):
            _delta.output_data()
            _delta._save_iter += 1

        _delta.make_figure.call_count == 5
        _delta.save_figure.call_count == 5
        _delta.save_grids.call_count == 5

        nc_size_after = os.path.getsize(exp_path_nc)
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

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert not os.path.isfile(exp_path_nc)

        for _t in range(0, 5):
            _delta.output_data()
            _delta._save_iter += 1
            _delta._time += _delta._dt

        # assertions
        _delta.make_figure.assert_any_call('eta', 0)
        _delta.make_figure.assert_any_call('stage', 0)
        _delta.make_figure.assert_any_call('depth', 0)
        _delta.make_figure.assert_any_call('qw', 0)
        _delta.make_figure.assert_any_call('uw', 0)
        _delta.make_figure.assert_any_call('qs', 0)
        for _i in range(1, 5):
            _delta.make_figure.assert_any_call('eta', _delta._dt * _i)
        _delta.make_figure.call_count == 5 * 6
        _delta.save_figure.call_count == 5 * 6
        _delta.save_grids.call_count == 0

    def test_save_metadata_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_metadata': True})
        _delta = DeltaModel(input_file=p)

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 3):
            _delta.output_data()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
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

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 3):
            _delta.output_data()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
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

        # mock the log_info
        _delta.log_info = mock.MagicMock()

        # mock the actual output routines
        _delta.make_figure = mock.MagicMock()
        _delta.save_figure = mock.MagicMock()
        _delta.save_grids = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _t in range(0, 6):
            _delta.output_data()
            _delta._save_iter += 1

        # close the file and connect
        _delta.output_netcdf.close()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")

        # assertions
        _arr = ds.variables['eta']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too
        assert ds.groups['meta']['H_SL'].shape[0] == _arr.shape[0]
        assert np.all(ds.groups['meta']['C0_percent'][:] == 0.2)
        assert np.all(ds.groups['meta']['f_bedload'][:] == 0.5)


class TestSaveFigure:

    def test_save_figure(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False,
                                      'save_eta_figs': True})
        _delta = DeltaModel(input_file=p)

        # make a figure
        fig, ax = plt.subplots()
        ax.imshow(np.random.uniform(0, 1, size=(100, 100)))

        # save two figs with different timesteps
        _delta.save_figure(fig, directory=_delta.prefix,
                           filename_root='eta_',
                           timestep=0)

        _delta.save_figure(fig, directory=_delta.prefix,
                           filename_root='eta_',
                           timestep=1)

        # check for output eta file
        exp_path_png0 = os.path.join(tmp_path / 'out_dir', 'eta_00000.png')
        assert os.path.isfile(exp_path_png0)
        exp_path_png1 = os.path.join(tmp_path / 'out_dir', 'eta_00001.png')
        assert os.path.isfile(exp_path_png1)

    def test_save_figure_sequential(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False,
                                      'save_eta_figs': True,
                                      'save_figs_sequential': False})
        _delta = DeltaModel(input_file=p)

        # make a figure
        fig, ax = plt.subplots()
        ax.imshow(np.random.uniform(0, 1, size=(100, 100)))

        # save two figs with different timesteps
        _delta.save_figure(fig, directory=_delta.prefix,
                           filename_root='eta_',
                           timestep=0)

        _delta.save_figure(fig, directory=_delta.prefix,
                           filename_root='eta_',
                           timestep=1)

        exp_path_png0 = os.path.join(
            tmp_path / 'out_dir', 'eta_00000.png')
        exp_path_png0_latest = os.path.join(
            tmp_path / 'out_dir', 'eta_latest.png')
        assert not os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png0_latest)


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
