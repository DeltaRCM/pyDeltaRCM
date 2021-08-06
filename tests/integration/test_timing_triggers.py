import glob
import os
import netCDF4
import numpy as np

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from .. import utilities


class TestTimingOutputData:

    def test_update_make_record(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True})
        _delta = DeltaModel(input_file=p)

        # modify the save interval to be twice dt
        _delta._save_dt = 2 * _delta._dt
        _delta._checkpoint_dt = 2 * _delta._dt

        # mock top-level methods, verify call was made to each
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_info = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.save_the_checkpoint = mock.MagicMock()
        # DO NOT mock output_data our output_checkpoint

        # mock the calls inside output_data
        _delta.save_stratigraphy = mock.MagicMock()
        _delta.save_grids_and_figs = mock.MagicMock()

        # run the timestep: t=0
        #   * should call core, but nothing else after init
        _delta.update()

        # assert calls
        assert _delta.solve_water_and_sediment_timestep.call_count == 1
        assert _delta.apply_subsidence.call_count == 1
        assert _delta.finalize_timestep.call_count == 1
        assert _delta.log_model_time.call_count == 1

        # assert times / counters
        assert _delta.time_iter == int(1)
        assert _delta.time == _delta.dt
        assert _delta.save_time_since_data == _delta._dt
        assert _delta.save_iter == int(1)
        assert _delta._save_time_since_checkpoint == _delta._dt

        # run another step
        #   * should call core steps and outputs
        _delta.update()

        # assert calls
        assert _delta.solve_water_and_sediment_timestep.call_count == 2
        assert _delta.apply_subsidence.call_count == 2
        assert _delta.finalize_timestep.call_count == 2
        assert _delta.log_model_time.call_count == 2

        # assert times / counters
        assert _delta.time_iter == int(2)
        assert _delta.time == 2 * _delta.dt
        assert _delta.save_time_since_data == 0
        assert _delta.save_iter == int(2)
        assert _delta._save_time_since_checkpoint == 0

        # run another step
        #   should call core, but nothing else
        _delta.update()

        # assert calls
        assert _delta.solve_water_and_sediment_timestep.call_count == 3
        assert _delta.apply_subsidence.call_count == 3
        assert _delta.finalize_timestep.call_count == 3
        assert _delta.log_model_time.call_count == 3

        # assert times / counters
        assert _delta.time_iter == int(3)
        assert _delta.time == 3 * _delta.dt
        assert _delta.save_time_since_data == _delta._dt
        assert _delta.save_iter == int(2)
        assert _delta._save_time_since_checkpoint == _delta._dt

    def test_update_saving_intervals_on_cycle(self, tmp_path):
        """dt == 300; save_dt == 600"""
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # modify the timestep and save interval to be twice dt
        _delta._dt = 300
        _delta._save_dt = 2 * _delta._dt
        _delta._save_any_grids = True  # override from settings

        # mock top-level methods, verify call was made to each
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()
        # DO NOT mock output_data

        # mock the calls inside output_data
        _delta.save_grids_and_figs = mock.MagicMock()

        _delta.update()  # no new saves, after init

        assert _delta.time_iter == int(1)
        assert _delta.time == _delta.dt
        assert _delta.save_iter == 1
        assert _delta.save_time_since_data == _delta.dt

        _delta.update()  # saves now

        assert _delta.time_iter == int(2)
        assert _delta.time == 2 * _delta.dt
        assert _delta.save_iter == 2
        assert _delta.save_time_since_data == 0

        _delta.update()  # no saves

        assert _delta.time_iter == int(3)
        assert _delta.save_iter == 2
        assert _delta.save_time_since_data == _delta.dt

        # run for a few to bring iters to 10
        for _ in range(7):
            _delta.update()

        assert _delta.time_iter == int(10)
        assert _delta.save_iter == 6

        assert _delta.time == 3000
        assert _delta._is_finalized is False

    def test_update_saving_intervals_short(self, tmp_path):
        """dt == 300; save_dt == 100"""
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # modify the timestep and save interval to be twice dt
        _delta._dt = 300
        _delta._save_dt = 100
        _delta._save_any_grids = True  # override from settings

        # mock top-level methods, verify call was made to each
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()
        # DO NOT mock output_data

        # mock the calls inside output_data
        _delta.save_grids_and_figs = mock.MagicMock()

        _delta.update()  # save on first iteration

        assert _delta.time_iter == int(1)
        assert _delta.time == _delta.dt
        assert _delta.save_grids_and_figs.call_count == 1

        _delta.update()  # save again

        assert _delta.save_time_since_data == 0
        assert _delta.time_iter == int(2)
        assert _delta.time == 2 * _delta.dt
        assert _delta.save_grids_and_figs.call_count == 2

        _delta.update()  # save again

        assert _delta.save_time_since_data == 0
        assert _delta.time_iter == int(3)
        assert _delta.save_grids_and_figs.call_count == 3

        assert _delta.save_iter == 4  # once during init
        assert _delta.time == 900

    def test_update_saving_intervals_offset_long_not_double(self, tmp_path):
        """dt == 300; save_dt == 500"""
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # modify the timestep and save interval to be twice dt
        _delta._dt = 300
        _delta._save_dt = 500
        _delta._save_any_grids = True  # override from settings

        # mock top-level methods, verify call was made to each
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()
        # DO NOT mock output_data

        # mock the calls inside output_data
        _delta.save_grids_and_figs = mock.MagicMock()

        _delta.update()

        assert _delta.time_iter == int(1)
        assert _delta.time == _delta.dt
        assert _delta.save_grids_and_figs.call_count == 0

        _delta.update()

        assert _delta.time_iter == int(2)
        assert _delta.time == 2 * _delta.dt
        assert _delta.save_grids_and_figs.call_count == 1

        for _ in range(10):
            _delta.update()

        assert _delta.time_iter == int(12)
        assert _delta.time == 3600
        assert _delta.save_grids_and_figs.call_count == 6

    def test_update_saving_intervals_offset_long_over_double(self, tmp_path):
        """dt == 300; save_dt == 1000"""
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # modify the timestep and save interval to be twice dt
        _delta._dt = 300
        _delta._save_dt = 1000
        _delta._save_any_grids = True  # override from settings

        # mock top-level methods, verify call was made to each
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()
        # DO NOT mock output_data

        # mock the calls inside output_data
        _delta.save_grids_and_figs = mock.MagicMock()

        _delta.update()

        assert _delta.time_iter == int(1)
        assert _delta.time == _delta.dt

        _delta.update()

        assert _delta.time_iter == int(2)
        assert _delta.time == 2 * _delta.dt
        assert _delta.save_grids_and_figs.call_count == 0

        _delta.update()

        assert _delta.time_iter == int(3)
        assert _delta.time == 3 * _delta.dt
        assert _delta.save_grids_and_figs.call_count == 0

        _delta.update()

        assert _delta.time == 4 * _delta.dt
        assert _delta.save_grids_and_figs.call_count == 1

        _delta.update()

        assert _delta.time == 5 * _delta.dt
        assert _delta.save_grids_and_figs.call_count == 1

        for _ in range(33):
            _delta.update()

        assert _delta.time == 38 * _delta.dt
        assert _delta.save_grids_and_figs.call_count == 9
        assert _delta._is_finalized is False

    def test_finalize_updated(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock the top-level
        _delta.log_info = mock.MagicMock()
        _delta.output_data = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()

        # modify the save interval
        _t = 5
        _delta._save_dt = _t * _delta._dt
        _delta._checkpoint_dt = _t * _delta._dt

        # run a mock update / save
        _delta._time = _t * _delta._dt
        _delta._save_iter += int(1)
        _delta._save_time_since_data = 0
        _delta._save_time_since_checkpoint = 0

        # run finalize
        _delta.finalize()

        # assert calls
        #   should only hit top-levels
        assert _delta.log_info.call_count == 2
        assert _delta.output_data.call_count == 0
        assert _delta.output_checkpoint.call_count == 0

        assert _delta._is_finalized is True

    def test_save_one_fig_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False,
                                      'save_eta_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()

        # check one set images created during init
        img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
        nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
        assert len(img_glob) == 1
        assert len(nc_glob) == 0

        # update the delta a few times
        for _ in range(0, 2):
            _delta.update()

        assert _delta.time_iter == 2.0
        _delta.solve_water_and_sediment_timestep.call_count == 2

        # check for output eta files
        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'out_dir', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'out_dir', 'eta_00001.png')
        exp_path_png2 = os.path.join(tmp_path / 'out_dir', 'eta_00002.png')
        exp_path_png3 = os.path.join(tmp_path / 'out_dir', 'eta_00003.png')
        assert not os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)
        assert os.path.isfile(exp_path_png2)
        assert not os.path.isfile(exp_path_png3)

    def test_save_one_fig_one_grid(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_eta_grids': True,
                                      'save_discharge_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()

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

        # now finalize, and file size should stay the same
        _delta.finalize()
        nc_size_after = os.path.getsize(exp_path_nc)
        assert _delta.time_iter == 2.0
        assert nc_size_after == nc_size_middle
        assert nc_size_after > nc_size_before

    def test_save_metadata_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_metadata': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()

        assert _delta.time_iter == 2.0
        _delta.solve_water_and_sediment_timestep.call_count == 2

        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        assert not ('eta' in ds.variables)
        assert ds['meta']['H_SL'].shape[0] == 3
        assert ds['meta']['L0'][:] == 3

    def test_save_subsidence_metadata_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'toggle_subsidence': True,
                                      'start_subsidence': 0,
                                      'subsidence_rate': 1,
                                      'save_metadata': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()

        assert _delta.time_iter == 2.0
        _delta.solve_water_and_sediment_timestep.call_count == 2

        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        assert not ('eta' in ds.variables)
        assert ds['meta']['H_SL'].shape[0] == 3
        assert ds['meta']['L0'][:] == 3
        assert ds['meta']['sigma'].shape == _delta.sigma.shape
        assert np.all(ds['meta']['sigma'] == _delta.sigma)
        assert ds['meta']['start_subsidence'][:] == 0

    def test_save_one_grid_metadata_by_default(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_metadata': False,
                                      'save_eta_grids': True,
                                      'C0_percent': 0.2})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.solve_water_and_sediment_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 6):
            _delta.update()

        assert _delta.time_iter == 6.0
        _delta.solve_water_and_sediment_timestep.call_count == 6

        _delta.finalize()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['eta']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too
        assert ds.groups['meta']['H_SL'].shape[0] == _arr.shape[0]
        assert np.all(ds.groups['meta']['C0_percent'][:] == 0.2)
        assert np.all(ds.groups['meta']['f_bedload'][:] == 0.5)
