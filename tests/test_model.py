# unit tests for deltaRCM_driver.py

import pytest

import sys
import os
import numpy as np

import netCDF4

from .utilities import test_DeltaModel

# need to create a simple case of pydeltarcm object to test these functions


def test_init(test_DeltaModel):
    """
    test the deltaRCM_driver init (happened when delta.initialize was run)
    """
    assert test_DeltaModel.time_iter == 0.
    assert test_DeltaModel._is_finalized is False


def test_update(test_DeltaModel):
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(1)
    assert test_DeltaModel.time == test_DeltaModel.dt
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(2)
    assert test_DeltaModel.time == 2 * test_DeltaModel.dt
    assert test_DeltaModel._is_finalized is False


def get_saved_times_from_file(_path):
    """Utility for extracting saved times from netcdf file after closed."""
    exp_path_nc = os.path.join(os.path.join(_path, 'pyDeltaRCM_output.nc'))
    assert os.path.isfile(exp_path_nc)
    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    return ds.variables['time']


def test_update_saving_intervals_on_cycle(test_DeltaModel):
    """dt == 300; save_dt == 600"""
    test_DeltaModel.save_dt = 600
    test_DeltaModel._save_any_grids = True  # override from settings
    assert test_DeltaModel.strata_counter == 0
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(1)
    assert test_DeltaModel.time == test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1
    test_DeltaModel.update()
    assert test_DeltaModel.save_time_since_last == 600
    assert test_DeltaModel.time_iter == int(2)
    assert test_DeltaModel.time == 2 * test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1  # not saved yet
    test_DeltaModel.update()  # should save now, at top of update()
    assert test_DeltaModel.time_iter == int(3)
    assert test_DeltaModel.strata_counter == 2
    assert test_DeltaModel.save_iter == 2
    for _ in range(7):
        test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(10)
    assert test_DeltaModel.strata_counter == 5
    assert test_DeltaModel.save_iter == 5
    # time will be longer than number of saves.
    _saves = np.array(test_DeltaModel.output_netcdf.variables['time'])
    assert np.all(_saves == np.array([0, 600, 1200, 1800, 2400]))
    assert test_DeltaModel.time == 3000
    assert test_DeltaModel._is_finalized is False
    test_DeltaModel.finalize()
    with pytest.raises(RuntimeError):
        _saves = np.array(test_DeltaModel.output_netcdf.variables['time'])
    _saves = get_saved_times_from_file(test_DeltaModel.prefix_abspath)
    assert np.all(_saves == np.array([0, 600, 1200, 1800, 2400, 3000]))


def test_update_saving_intervals_short(test_DeltaModel):
    """dt == 300; save_dt == 100"""
    test_DeltaModel.save_dt = 100
    test_DeltaModel._save_any_grids = True  # override from settings
    assert test_DeltaModel.strata_counter == 0
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(1)
    assert test_DeltaModel.time == test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1
    test_DeltaModel.update()
    assert test_DeltaModel.save_time_since_last == 300
    assert test_DeltaModel.time_iter == int(2)
    assert test_DeltaModel.time == 2 * test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 2
    test_DeltaModel.update()
    assert test_DeltaModel.save_time_since_last == 300
    assert test_DeltaModel.time_iter == int(3)
    assert test_DeltaModel.strata_counter == 3
    assert test_DeltaModel.save_iter == 3
    assert test_DeltaModel.time == 900
    _saves = np.array(test_DeltaModel.output_netcdf.variables['time'])
    assert np.all(_saves == np.array([0, 300, 600]))
    test_DeltaModel.finalize()
    _saves = get_saved_times_from_file(test_DeltaModel.prefix_abspath)
    # new save, because 900, but not saved
    assert np.all(_saves == np.array([0, 300, 600, 900]))


def test_update_saving_intervals_offset_long_not_double(test_DeltaModel):
    """dt == 300; save_dt == 500"""
    test_DeltaModel.save_dt = 500
    test_DeltaModel._save_any_grids = True  # override from settings
    assert test_DeltaModel.strata_counter == 0
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(1)
    assert test_DeltaModel.time == test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(2)
    assert test_DeltaModel.time == 2 * test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1
    for _ in range(10):
        test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(12)
    assert test_DeltaModel.time == 3600
    assert test_DeltaModel.strata_counter == 6
    _saves = np.array(test_DeltaModel.output_netcdf.variables['time'])
    assert np.all(_saves == np.array([0, 600, 1200, 1800, 2400, 3000]))
    test_DeltaModel.finalize()
    _saves = get_saved_times_from_file(test_DeltaModel.prefix_abspath)
    # add final save, because time reached
    assert np.all(_saves == np.array([0, 600, 1200, 1800, 2400, 3000, 3600]))


def test_update_saving_intervals_offset_long_over_double(test_DeltaModel):
    """dt == 300; save_dt == 1000"""
    test_DeltaModel.save_dt = 1000
    test_DeltaModel._save_any_grids = True  # override from settings
    assert test_DeltaModel.strata_counter == 0
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(1)
    assert test_DeltaModel.time == test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(2)
    assert test_DeltaModel.time == 2 * test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1
    test_DeltaModel.update()
    assert test_DeltaModel.time_iter == int(3)
    assert test_DeltaModel.time == 3 * test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1
    test_DeltaModel.update()
    assert test_DeltaModel.time == 4 * test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 1
    test_DeltaModel.update()
    assert test_DeltaModel.time == 5 * test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 2
    for _ in range(33):
        test_DeltaModel.update()
    assert test_DeltaModel.time == 38 * test_DeltaModel.dt
    assert test_DeltaModel.strata_counter == 10
    assert test_DeltaModel._is_finalized is False
    _saves = np.array(test_DeltaModel.output_netcdf.variables['time'])
    assert np.all(_saves == np.array([0, 1200, 2400, 3600, 4800, 6000, 7200, 8400, 9600, 10800]))
    test_DeltaModel.finalize()
    _saves = get_saved_times_from_file(test_DeltaModel.prefix_abspath)
    # no new saves, because time not reached
    assert np.all(_saves == np.array([0, 1200, 2400, 3600, 4800, 6000, 7200, 8400, 9600, 10800]))


def test_finalize(test_DeltaModel):
    for _ in range(2):
        test_DeltaModel.update()
    test_DeltaModel.finalize()
    assert test_DeltaModel._is_finalized is True


def test_output_strata_error_if_no_updates(test_DeltaModel):
    with pytest.raises(RuntimeError, match=r'Model has no computed strat.*'):
        test_DeltaModel.output_strata()


def test_multifinalization_error(test_DeltaModel):
    err_delta = test_DeltaModel
    assert err_delta.dt == 300.0
    err_delta.save_dt = 300.0
    err_delta.update()
    # test will fail if any assertion is wrong
    assert err_delta.time_iter == 1.0
    assert err_delta._is_finalized is False
    err_delta.finalize()
    assert err_delta._is_finalized is True
    # next line should throw RuntimeError
    with pytest.raises(RuntimeError, match=r'Cannot update model,.*'):
        err_delta.update()


def test_initial_values(test_DeltaModel):
    assert np.all(test_DeltaModel.sea_surface_elevation == 0)
    assert test_DeltaModel.water_depth[0, 2] == 0
    assert test_DeltaModel.water_depth[0, 3] == 1
    assert test_DeltaModel.water_depth[4, 4] == 1
    assert test_DeltaModel.bed_elevation[0, 2] == 0
    assert test_DeltaModel.bed_elevation[0, 3] == -1
    assert test_DeltaModel.bed_elevation[4, 4] == -1


def test_setting_getting_sea_surface_mean_elevation(test_DeltaModel):
    assert test_DeltaModel.sea_surface_mean_elevation == 0
    test_DeltaModel.sea_surface_mean_elevation = 0.5
    assert test_DeltaModel.sea_surface_mean_elevation == 0.5


def test_setting_getting_sea_surface_elevation_change(test_DeltaModel):
    assert test_DeltaModel.sea_surface_elevation_change == 0.001
    test_DeltaModel.sea_surface_elevation_change = 0.002
    assert test_DeltaModel.sea_surface_elevation_change == 0.002


def test_setting_getting_bedload_fraction(test_DeltaModel):
    assert test_DeltaModel.bedload_fraction == 0.5
    test_DeltaModel.bedload_fraction = 0.25
    assert test_DeltaModel.bedload_fraction == 0.25


def test_setting_getting_channel_flow_velocity(test_DeltaModel):
    assert test_DeltaModel.channel_flow_velocity == 1
    test_DeltaModel.channel_flow_velocity = 3
    assert test_DeltaModel.channel_flow_velocity == 3
    assert test_DeltaModel.u_max == 6


def test_setting_getting_channel_width(test_DeltaModel):
    assert test_DeltaModel.channel_width == 2
    test_DeltaModel.channel_width = 10
    assert test_DeltaModel.channel_width == 10
    assert test_DeltaModel.N0 == 10


def test_setting_getting_channel_width(test_DeltaModel):
    assert test_DeltaModel.channel_flow_depth == 1
    test_DeltaModel.channel_flow_depth = 2
    assert test_DeltaModel.channel_flow_depth == 2


def test_setting_getting_channel_width(test_DeltaModel):
    assert test_DeltaModel.influx_sediment_concentration == 0.1
    test_DeltaModel.influx_sediment_concentration = 2
    assert test_DeltaModel.C0 == 0.02


def test_make_checkpoint(tmp_path, test_DeltaModel):
    """Test setting the checkpoint option to 'True' and saving a checkpoint."""
    test_DeltaModel.save_checkpoint = True
    test_DeltaModel._save_checkpoint = True
    test_DeltaModel.checkpoint_dt = 1
    test_DeltaModel.save_dt = 1
    check_DeltaModel = test_DeltaModel
    test_DeltaModel.update()
    exp_path_npz = os.path.join(tmp_path / 'out_dir', 'checkpoint.npz')
    assert os.path.isfile(exp_path_npz)
    test_DeltaModel.finalize()
    # check loading checkpoint and see if it works
    check_DeltaModel.load_checkpoint()
    assert check_DeltaModel._time == test_DeltaModel._time
    assert np.all(check_DeltaModel.uw == test_DeltaModel.uw)
    assert np.all(check_DeltaModel.ux == test_DeltaModel.ux)
    assert np.all(check_DeltaModel.uy == test_DeltaModel.uy)
    assert np.all(check_DeltaModel.depth == test_DeltaModel.depth)
    assert np.all(check_DeltaModel.stage == test_DeltaModel.stage)
    assert np.all(check_DeltaModel.eta == test_DeltaModel.eta)
    assert np.all(check_DeltaModel.strata_eta.todense() ==
                  test_DeltaModel.strata_eta.todense())
    assert np.all(check_DeltaModel.strata_sand_frac.todense() ==
                  test_DeltaModel.strata_sand_frac.todense())
