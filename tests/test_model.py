# unit tests for deltaRCM_driver.py

import numpy as np

import pytest

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM import shared_tools
from . import utilities


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

    def test_error_if_no_file_found(self):
        with pytest.raises(FileNotFoundError):
            _ = DeltaModel(input_file='./nonexisting_file.yaml')

    def test_override_single_default(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'S0', 0.005)
        f.close()
        delta = DeltaModel(input_file=p)
        assert delta.S0 == 0.005

    def test_override_two_defaults(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'S0', 0.005)
        utilities.write_parameter_to_file(f, 'Np_sed', 2)
        f.close()
        delta = DeltaModel(input_file=p)
        assert delta.S0 == 0.005
        assert delta.Np_sed == 2

    def test_override_single_default_kwarg(self, tmp_path):
        _out_dir = str(tmp_path / 'out_dir')
        delta = DeltaModel(out_dir=_out_dir, S0=0.005)
        assert delta.S0 == 0.005

    def test_override_two_defaults_kwargs(self, tmp_path):
        _out_dir = str(tmp_path / 'out_dir')
        delta = DeltaModel(
            out_dir=_out_dir,
            S0=0.005,
            Np_sed=2)
        assert delta.S0 == 0.005
        assert delta.Np_sed == 2

    def test_override_yaml_with_kwarg(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'S0', 0.005)
        f.close()
        with pytest.warns(UserWarning):
            delta = DeltaModel(input_file=p, S0=0.0333)
        assert delta.S0 == 0.0333

    def test_override_bad_type_float_string(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'S0', 'a string?!')
        f.close()
        with pytest.raises(TypeError):
            _ = DeltaModel(input_file=p)

    def test_override_bad_type_int_float(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'beta', 24.4234)
        f.close()
        with pytest.raises(TypeError):
            _ = DeltaModel(input_file=p)

    def test_not_creating_illegal_attributes(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'illegal_attribute', True)
        f.close()
        delta = DeltaModel(input_file=p)
        assert delta.S0 == 0.00015  # from default.yaml
        assert not hasattr(delta, 'illegal_attribute')

    def test_not_overwriting_existing_attributes(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'input_file', '/fake/path.yaml')
        f.close()
        delta = DeltaModel(input_file=p)
        assert delta.S0 == 0.00015  # from default.yaml
        assert hasattr(delta, 'input_file')
        assert delta.input_file == p

    def test_random_seed_settings_value(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'seed', 9999)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        f.close()
        shared_tools.set_random_seed(9999)
        _preval_same = shared_tools.get_random_uniform(1)
        shared_tools.set_random_seed(5)
        _preval_diff = shared_tools.get_random_uniform(1)
        delta = DeltaModel(input_file=p)
        assert delta.seed == 9999
        _postval_same = shared_tools.get_random_uniform(1)
        assert _preval_same == _postval_same
        assert delta.seed == 9999

    def test_random_seed_settings_newinteger_default(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'S0', 0.005)
        f.close()
        delta = DeltaModel(input_file=p)
        assert delta.seed is not None
        assert delta.seed <= (2**32) - 1
        assert isinstance(int(delta.seed), int)

    def test_negative_length(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Length', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_width(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Width', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_dx(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'dx', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_bigger_than_Width_dx(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Width', 10)
        utilities.write_parameter_to_file(f, 'dx', 100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_bigger_than_Length_dx(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Length', 10)
        utilities.write_parameter_to_file(f, 'dx', 100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_L0_meters(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'L0_meters', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_itermax(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'itermax', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_Np_water(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Np_water', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_N0_meters(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'N0_meters', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_Np_sed(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Np_sed', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_f_bedload(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'f_bedload', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_big_f_bedload(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'f_bedload', 2)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_C0_percent(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'C0_percent', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)

    def test_negative_Csmooth(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Csmooth', -100)
        f.close()
        with pytest.raises(ValueError):
            _ = DeltaModel(input_file=p)


class TestUpdate:

    def test_update(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock top-level methods, verify call was made to each
        _delta.run_one_timestep = mock.MagicMock()
        _delta.apply_subsidence = mock.MagicMock()
        _delta.finalize_timestep = mock.MagicMock()
        _delta.log_model_time = mock.MagicMock()
        _delta.output_data = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()

        # run the timestep: t=0
        #   * should call all methods
        _delta.update()

        # assert calls
        assert _delta.run_one_timestep.call_count == 1
        assert _delta.apply_subsidence.call_count == 1
        assert _delta.finalize_timestep.call_count == 1
        assert _delta.log_model_time.call_count == 1
        assert _delta.output_data.call_count == 1
        assert _delta.output_checkpoint.call_count == 1

        # assert times / counters
        assert _delta.time_iter == int(1)
        assert _delta.time == _delta.dt

        # run another step
        #   * should call all steps again
        _delta.update()

        # assert calls
        assert _delta.run_one_timestep.call_count == 2
        assert _delta.apply_subsidence.call_count == 2
        assert _delta.finalize_timestep.call_count == 2
        assert _delta.log_model_time.call_count == 2
        assert _delta.output_data.call_count == 2
        assert _delta.output_checkpoint.call_count == 2

        # assert times / counters
        assert _delta.time_iter == int(2)
        assert _delta.time == 2 * _delta.dt

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
        _delta.output_data = mock.MagicMock()
        _delta.output_checkpoint = mock.MagicMock()
        _delta.record_final_stratigraphy = mock.MagicMock()

        # run finalize
        _delta.finalize()

        # assert calls
        #  should hit all options since no saves
        assert _delta.log_info.call_count == 2
        assert _delta.record_final_stratigraphy.call_count == 1

        # these were originally included in `finalize`, but no longer.
        #   the checks for no call are here to ensure we don't revert
        assert _delta.output_data.call_count == 0
        assert _delta.output_checkpoint.call_count == 0

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
