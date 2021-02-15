# unit tests for deltaRCM_driver.py

import os
import numpy as np

import pytest

import netCDF4

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from . import utilities
from .utilities import test_DeltaModel

# need to create a simple case of pydeltarcm object to test these functions


class Test__init__:

    def test_init(self, tmp_path):
        """
        test the deltaRCM_driver init (happened when delta.initialize was run)
        """
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        assert _delta.time_iter == 0.
        assert _delta._is_finalized is False


class TestUpdate:

    def test_update_make_record(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # modify the save interval to be twice dt
        _delta._save_dt = 2 * _delta._dt
        _delta._checkpoint_dt = 2 * _delta._dt

        # mock top-level methods, verify call was made to each
        _delta.record_stratigraphy = mock.MagicMock()
        _delta.output_data = mock.MagicMock()
        _delta.run_one_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()

        # run the timestep: t=0
        #   * should call output and core
        #   * does not output checkpoint on t=0
        _delta.update()

        # assert calls
        assert _delta.record_stratigraphy.call_count == 1
        assert _delta.output_data.call_count == 1
        assert _delta.run_one_timestep.call_count == 1
        assert _delta.apply_subsidence.call_count == 1
        assert _delta.finalize_timestep.call_count == 1
        assert _delta.log_model_time.call_count == 1
        assert _delta.output_checkpoint.call_count == 0

        # assert times / counters
        assert _delta.time_iter == int(1)
        assert _delta.time == _delta.dt
        assert _delta.save_time_since_last == _delta._dt
        assert _delta.save_iter == int(1)
        assert _delta._save_time_since_checkpoint == _delta._dt

        # run another step
        #   * should only call core steps
        _delta.update()

        # assert calls
        assert _delta.record_stratigraphy.call_count == 1
        assert _delta.output_data.call_count == 1
        assert _delta.run_one_timestep.call_count == 2
        assert _delta.apply_subsidence.call_count == 2
        assert _delta.finalize_timestep.call_count == 2
        assert _delta.log_model_time.call_count == 2
        assert _delta.output_checkpoint.call_count == 0

        # assert times / counters
        assert _delta.time_iter == int(2)
        assert _delta.time == 2 * _delta.dt
        assert _delta.save_time_since_last == 2 * _delta._dt
        assert _delta.save_iter == int(1)
        assert _delta._save_time_since_checkpoint == 2 * _delta._dt

        # run another step
        #   should call output, core, and checkpoint steps
        _delta.update()

        # assert calls
        assert _delta.record_stratigraphy.call_count == 2
        assert _delta.output_data.call_count == 2
        assert _delta.run_one_timestep.call_count == 3
        assert _delta.apply_subsidence.call_count == 3
        assert _delta.finalize_timestep.call_count == 3
        assert _delta.log_model_time.call_count == 3
        assert _delta.output_checkpoint.call_count == 1

        # assert times / counters
        assert _delta.time_iter == int(3)
        assert _delta.time == 3 * _delta.dt
        assert _delta.save_time_since_last == _delta._dt
        assert _delta.save_iter == int(2)
        assert _delta._save_time_since_checkpoint == _delta._dt

    def test_update_is_finalized(self, tmp_path):
        # create a delta with different itermax
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # change state to finalized
        _delta._is_finalized = True

        # run the timestep
        with pytest.raises(RuntimeError):
            _delta.update()


class TestFinalize:

    def test_finalize_not_updated(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        _delta.log_info = mock.MagicMock()
        _delta.record_stratigraphy = mock.MagicMock()
        _delta.output_data = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()
        _delta.output_strata = mock.MagicMock()

        # run finalize
        _delta.finalize()

        # assert calls
        #  should hit all options since no saves
        assert _delta.log_info.call_count == 2
        assert _delta.record_stratigraphy.call_count == 1
        assert _delta.output_data.call_count == 1
        assert _delta.output_checkpoint.call_count == 0
        assert _delta.output_strata.call_count == 1

        assert _delta._is_finalized is True

    def test_finalize_updated(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock the top-level
        _delta.log_info = mock.MagicMock()
        _delta.record_stratigraphy = mock.MagicMock()
        _delta.output_data = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()
        _delta.output_strata = mock.MagicMock()

        # modify the save interval
        _t = 5
        _delta._save_dt = _t * _delta._dt
        _delta._checkpoint_dt = _t * _delta._dt

        # run a mock update / save
        _delta._time = _t * _delta._dt
        _delta._save_iter += int(1)
        _delta._save_time_since_last = 0
        _delta._save_time_since_checkpoint = 0

        # run finalize
        _delta.finalize()

        # assert calls
        #   should only hit top-levels
        assert _delta.log_info.call_count == 2
        assert _delta.record_stratigraphy.call_count == 0
        assert _delta.output_data.call_count == 0
        assert _delta.output_checkpoint.call_count == 0
        assert _delta.output_strata.call_count == 1

        assert _delta._is_finalized is True

    def test_finalize_is_finalized(self, tmp_path):
        # create a delta with different itermax
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # change state to finalized
        _delta._is_finalized = True

        # run the timestep
        with pytest.raises(RuntimeError):
            _delta.finalize()

        assert _delta._is_finalized is True


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



def test_output_strata_error_if_no_updates(test_DeltaModel):
    with pytest.raises(RuntimeError, match=r'Model has no computed strat.*'):
        test_DeltaModel.output_strata()


class TestPublicSettersAndGetters:

    def test_setting_getting_sea_surface_mean_elevation(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        assert _delta.sea_surface_mean_elevation == 0
        _delta.sea_surface_mean_elevation = 0.5
        assert _delta.sea_surface_mean_elevation == 0.5

    def test_setting_getting_sea_surface_elevation_change(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        assert _delta.sea_surface_elevation_change == 0
        _delta.sea_surface_elevation_change = 0.002
        assert _delta.sea_surface_elevation_change == 0.002

    def test_setting_getting_bedload_fraction(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        assert _delta.bedload_fraction == 0.5
        _delta.bedload_fraction = 0.25
        assert _delta.bedload_fraction == 0.25

    def test_setting_getting_channel_flow_velocity(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock the init methods
        _delta.create_other_variables = mock.MagicMock()
        _delta.init_sediment_routers = mock.MagicMock()

        # check initials
        assert _delta.channel_flow_velocity == 1

        # change and assert changed
        _delta.channel_flow_velocity = 3
        assert _delta.channel_flow_velocity == 3
        assert _delta.channel_flow_velocity == _delta.u0
        assert _delta.channel_flow_velocity == _delta._u0

        # assert reinitializers called
        assert (_delta.create_other_variables.called is True)
        assert (_delta.init_sediment_routers.called is True)

    def test_setting_getting_channel_width(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        _delta.create_other_variables = mock.MagicMock()
        _delta.init_sediment_routers = mock.MagicMock()

        # check initials
        assert _delta.channel_width == 250

        # change value
        #  the channel width is then changed internally with `N0`, according
        #  to the `create_other_variables` so no change is actually made
        #  here.
        with pytest.warns(UserWarning):
            _delta.channel_width = 300
        assert _delta.channel_width == 250  # not changed!

        # assert reinitializers called
        assert (_delta.create_other_variables.called is True)
        assert (_delta.init_sediment_routers.called is True)

    def test_setting_getting_flow_depth(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock the init methods
        _delta.create_other_variables = mock.MagicMock()
        _delta.init_sediment_routers = mock.MagicMock()

        # check initials
        assert _delta.channel_flow_depth == 5

        # change and assert changed
        _delta.channel_flow_depth = 2
        assert _delta.channel_flow_depth == 2
        assert _delta.channel_flow_depth == _delta.h0
        assert _delta.channel_flow_depth == _delta._h0

        # assert reinitializers called
        assert (_delta.create_other_variables.called is True)
        assert (_delta.init_sediment_routers.called is True)

    def test_setting_getting_influx_concentration(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock the init methods
        _delta.create_other_variables = mock.MagicMock()
        _delta.init_sediment_routers = mock.MagicMock()

        # check initials
        assert _delta.influx_sediment_concentration == 0.1

        # change and assert changed
        _delta.influx_sediment_concentration = 0.003
        assert _delta.C0_percent == 0.3

        # assert reinitializers called
        assert (_delta.create_other_variables.called is True)
        assert (_delta.init_sediment_routers.called is True)

    def test_getter_nosetter_sea_surface_elevation(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # check that one is alias of other
        assert np.all(_delta.sea_surface_elevation == _delta.stage)
        assert (_delta.sea_surface_elevation is _delta.stage)

        # setter should return error, not allowed
        with pytest.raises(AttributeError):
            _delta.sea_surface_elevation = np.random.uniform(
                0, 1, size=_delta.eta.shape)

    def test_getter_nosetter_water_depth(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # check that one is alias of other
        assert np.all(_delta.water_depth == _delta.depth)
        assert (_delta.water_depth is _delta.depth)

        # setter should return error, not allowed
        with pytest.raises(AttributeError):
            _delta.water_depth = np.random.uniform(
                0, 1, size=_delta.eta.shape)

    def test_getter_nosetter_bed_elevation(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # check that one is alias of other
        assert np.all(_delta.bed_elevation == _delta.eta)
        assert (_delta.bed_elevation is _delta.eta)

        # setter should return error, not allowed
        with pytest.raises(AttributeError):
            _delta.bed_elevation = np.random.uniform(
                0, 1, size=_delta.eta.shape)
