# unit tests for init_tools.py

import pytest
import unittest.mock as mock

import numpy as np
import os
from netCDF4 import Dataset

from pyDeltaRCM.model import DeltaModel

from . import utilities


class TestModelDomainSetup:

    def test_inlet_size_specified(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Length', 4000.)
        utilities.write_parameter_to_file(f, 'Width', 8000.)
        utilities.write_parameter_to_file(f, 'dx', 20)
        utilities.write_parameter_to_file(f, 'N0_meters', 150)
        utilities.write_parameter_to_file(f, 'L0_meters', 200)
        f.close()
        delta = DeltaModel(input_file=p)
        assert delta.N0 == 8
        assert delta.L0 == 10

    def test_inlet_size_set_to_one_fourth_domain(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Length', 4000.)
        utilities.write_parameter_to_file(f, 'Width', 8000.)
        utilities.write_parameter_to_file(f, 'dx', 20)
        utilities.write_parameter_to_file(f, 'N0_meters', 5500)
        utilities.write_parameter_to_file(f, 'L0_meters', 3300)
        f.close()
        delta = DeltaModel(input_file=p)
        assert delta.N0 == 100
        assert delta.L0 == 50

    def test_x(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.x[0][-1] == 199

    def test_y(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.y[0][-1] == 0

    def test_cell_type(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        # wall type in corner
        assert delta.cell_type[0, 0] == -2

    def test_eta(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.eta[10, 10] == -5

    def test_stage(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.stage[10, 10] == 0.0

    def test_depth(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.depth[10, 10] == 5

    def test_qx(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        # prescribe the qx at the inlet
        assert delta.qx[0, delta.CTR] == 5

    def test_qy(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.qy[0, delta.CTR] == 0

    def test_qxn(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.qxn[0, 0] == 0

    def test_qyn(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.qyn[0, 0] == 0

    def test_qwn(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.qwn[0, 0] == 0

    def test_ux(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.ux[0, delta.CTR] == 1

    def test_uy(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.uy[0, delta.CTR] == 0

    def test_uw(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.uw[0, delta.CTR] == 1

    def test_qs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.qs[5, 5] == 0

    def test_Vp_dep_sand(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert np.any(delta.Vp_dep_sand) == 0

    def test_Vp_dep_mud(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert np.any(delta.Vp_dep_mud) == 0

    def test_free_surf_flag(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert np.any(delta.free_surf_flag) == 0

    def test_indices(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert np.any(delta.free_surf_walk_inds) == 0

    def test_sfc_visit(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert np.any(delta.sfc_visit) == 0

    def test_sfc_sum(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert np.any(delta.sfc_sum) == 0


class TestCreateBoundaryConditions:

    # base case during init is covered by tests elsewhere!

    def test_change_variable_updated_bcs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # what is Qw0 before?
        assert _delta.Qw0 == 1250.0

        # change u0
        _delta.u0 = 2

        # nothing has happened to Qw0 yet
        assert _delta.u0 == 2
        assert _delta.Qw0 == 1250.0

        # now call to recreate, see what happened
        _delta.create_boundary_conditions()
        assert _delta.u0 == 2
        assert _delta.Qw0 == 2500.0


class TestInitSubsidence:

    def test_subsidence_bounds(self, tmp_path):
        """Test subsidence bounds."""

        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 600.)
        utilities.write_parameter_to_file(f, 'Width', 600.)
        utilities.write_parameter_to_file(f, 'dx', 5)
        utilities.write_parameter_to_file(f, 'toggle_subsidence', True)
        f.close()

        delta = DeltaModel(input_file=p)
        # assert subsidence mask is binary
        assert np.all(delta.subsidence_mask ==
                      delta.subsidence_mask.astype(bool))
        # check specific regions
        assert np.all(delta.subsidence_mask[delta.L0:, :] == 1)
        assert np.all(delta.subsidence_mask[:delta.L0, :] == 0)


class TestLoadCheckpoint:

    @mock.patch('pyDeltaRCM.shared_tools.set_random_state')
    def test_load_standard_grid(self, patched, tmp_path):
        """Test that a run can be resumed when there are outputs.
        """
        # create one delta, just to have a checkpoint file
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)

        # make mocks
        _delta.log_info = mock.MagicMock()
        _delta.logger = mock.MagicMock()
        _delta.init_output_file = mock.MagicMock()

        # close the file so can be safely opened in load
        _delta.output_netcdf.close()

        # check checkpoint exists
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))

        # now mess up a field
        _eta0 = np.copy(_delta.eta)
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field
        assert np.all(_delta.eta == _rand_field)

        # now resume from the checkpoint to restore the field
        _delta.load_checkpoint()

        # check that fields match
        assert np.all(_delta.eta == _eta0)

        # assertions on function calls
        _call = [mock.call('Renaming old NetCDF4 output file', verbosity=2)]
        _delta.log_info.assert_has_calls(_call, any_order=True)
        _delta.logger.assert_not_called()
        _delta.init_output_file.assert_not_called()
        patched.assert_called()

    @mock.patch('pyDeltaRCM.shared_tools.set_random_state')
    def test_load_wo_netcdf_not_expected(self, patched, tmp_path):
        """
        Test that a checkpoint can be loaded when the load does not expect
        there to be any netcdf file.
        """
        # create one delta, just to have a checkpoint file
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True})
        _delta = DeltaModel(input_file=p)

        # make mocks
        _delta.log_info = mock.MagicMock()
        _delta.logger = mock.MagicMock()
        _delta.init_output_file = mock.MagicMock()

        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))
        assert not os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))

        # now mess up a field
        _eta0 = np.copy(_delta.eta)
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field
        assert np.all(_delta.eta == _rand_field)

        # now resume from the checkpoint to restore the field
        _delta.load_checkpoint()

        # check that fields match
        assert np.all(_delta.eta == _eta0)

        # assertions on function calls
        _delta.log_info.assert_called()
        _delta.logger.assert_not_called()
        _delta.init_output_file.assert_not_called()
        patched.assert_called()

    @mock.patch('pyDeltaRCM.shared_tools.set_random_state')
    def test_load_wo_netcdf_expected(self, patched, tmp_path):
        """
        Test that a checkpoint can be loaded when the load expects there to be
        a netcdf file. This will create a new netcdf file and raise a
        warning.
        """
        # define a yaml with NO outputs, but checkpoint
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)

        # make mocks
        _delta.log_info = mock.MagicMock()
        _delta.logger = mock.MagicMock()
        _delta.init_output_file = mock.MagicMock()

        # close the file so can be safely opened in load
        _delta.output_netcdf.close()

        # check that files exist, and then delete nc
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))
        os.remove(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))

        # now mess up a field
        _eta0 = np.copy(_delta.eta)
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field
        assert np.all(_delta.eta == _rand_field)

        # now resume from the checkpoint to restore the field
        with pytest.warns(UserWarning, match=r'NetCDF4 output *.'):
            _delta.load_checkpoint()

        # check that fields match
        assert np.all(_delta.eta == _eta0)
        assert _delta._save_iter == 0

        # assertions on function calls
        _delta.log_info.assert_called()
        _delta.logger.warning.assert_called()
        _delta.init_output_file.assert_called()
        patched.assert_called()

    @mock.patch('pyDeltaRCM.shared_tools.set_random_state')
    def test_load_already_open_netcdf_error(self, patched, tmp_path):
        """
        Test that a checkpoint can be loaded when the load expects there to be
        a netcdf file. This will create a new netcdf file and raise a
        warning.
        """
        # define a yaml with an output, and checkpoint
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)

        # make mocks
        _delta.log_info = mock.MagicMock()
        _delta.logger = mock.MagicMock()
        _delta.init_output_file = mock.MagicMock()

        # close the file so can be safely opened in load
        _delta.output_netcdf.close()

        # check that files exist, and then open the nc back up
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))
        _ = Dataset(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))

        # now try to resume a model and should throw error
        with pytest.raises(RuntimeError):
            _delta.load_checkpoint()


class TestSettingConstants:
    """
    tests for all of the constants
    """

    def test_set_constant_g(self, tmp_path):
        """
        check gravity
        """
        # create delta from default options
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.g == 9.81

    def test_set_constant_distances(self, tmp_path):
        """
        check distances
        """
        # create delta from default options
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.distances[0, 0] == pytest.approx(np.sqrt(2))

    def test_set_ivec(self, tmp_path):
        # create delta from default options
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.ivec[0, 0] == pytest.approx(-np.sqrt(0.5))

    def test_set_jvec(self, tmp_path):
        # create delta from default options
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.jvec[0, 0] == pytest.approx(-np.sqrt(0.5))

    def test_set_iwalk(self, tmp_path):
        # create delta from default options
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.iwalk[0, 0] == -1

    def test_set_jwalk(self, tmp_path):
        # create delta from default options
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.jwalk[0, 0] == -1

    def test_kernel1(self, tmp_path):
        # create delta from default options
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.kernel1[0, 0] == 1

    def test_kernel2(self, tmp_path):
        # create delta from default options
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.kernel2[0, 0] == 1


class TestSettingParametersFromYAMLFile:
    # tests for attrs set during yaml parsing

    def test_init_verbose_default(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)
        assert delta.verbose == 0

    def test_init_verbose(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'verbose': 1})
        delta = DeltaModel(input_file=p)
        assert delta.verbose == 1

    def test_init_seed_zero(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'seed': 0})
        delta = DeltaModel(input_file=p)
        assert delta.seed == 0

    def test_init_Np_water(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'Np_water': 50})
        _delta = DeltaModel(input_file=p)
        assert _delta.init_Np_water == 50

    def test_init_Np_sed(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'Np_sed': 60})
        _delta = DeltaModel(input_file=p)
        assert _delta.init_Np_sed == 60

    def test_dx(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'dx': 20})
        _delta = DeltaModel(input_file=p)
        assert _delta.dx == 20

    def test_itermax(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'itermax': 6})
        _delta = DeltaModel(input_file=p)
        assert _delta.itermax == 6

    def test_h0(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'h0': 7.5})
        _delta = DeltaModel(input_file=p)
        assert _delta.h0 == 7.5

        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'h0': int(7)})
        _delta = DeltaModel(input_file=p)
        assert _delta.h0 == 7

    def test_hb(self, tmp_path):
        # take default from h0 if not given:
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'h0': 7.5})
        _delta = DeltaModel(input_file=p)
        assert _delta.h0 == 7.5
        assert _delta.hb == 7.5

        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'hb': 7.5})
        _delta = DeltaModel(input_file=p)
        assert _delta.h0 == 5
        assert _delta.hb == 7.5

        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'hb': int(7)})
        _delta = DeltaModel(input_file=p)
        assert _delta.h0 == 5
        assert _delta.hb == 7

    def test_Nsmooth(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'Nsmooth': 6})
        _delta = DeltaModel(input_file=p)
        assert _delta.Nsmooth == 6

    def test_SLR(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'SLR': 0.01})
        _delta = DeltaModel(input_file=p)
        assert _delta.SLR == 0.01

    def test_omega_flow(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'omega_flow': 0.8})
        _delta = DeltaModel(input_file=p)
        assert _delta.omega_flow == 0.8

    def test_lambda(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'sed_lag': 0.8})
        _delta = DeltaModel(input_file=p)
        assert _delta._lambda == 0.8

    def test_alpha(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'alpha': 0.25})
        _delta = DeltaModel(input_file=p)
        assert _delta.alpha == 0.25

    def test_stepmax_default(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'alpha': 0.25})
        _delta = DeltaModel(input_file=p)
        assert _delta.stepmax == 2 * (_delta.L + _delta.W)

    def test_stepmax_integer(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'stepmax': 10})
        _delta = DeltaModel(input_file=p)
        assert _delta.stepmax == 10

    def test_stepmax_float(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'stepmax': 11.0})
        _delta = DeltaModel(input_file=p)
        assert _delta.stepmax == 11

    def test_save_eta_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert 'eta' in _delta._save_var_list.keys()

    def test_save_depth_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_depth_grids': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert 'depth' in _delta._save_var_list.keys()

    def test_save_stage_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_stage_grids': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert 'stage' in _delta._save_var_list.keys()

    def test_save_discharge_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_discharge_grids': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert 'discharge' in _delta._save_var_list.keys()

    def test_save_velocity_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_velocity_grids': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert 'velocity' in _delta._save_var_list.keys()

    def test_save_sedflux_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_sedflux_grids': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert 'sedflux' in _delta._save_var_list.keys()

    def test_save_sandfrac_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_sandfrac_grids': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert 'sandfrac' in _delta._save_var_list.keys()

    def test_save_discharge_components(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_discharge_components': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert _delta._save_discharge_components is True
        assert 'discharge_x' in _delta._save_var_list.keys()
        assert 'discharge_y' in _delta._save_var_list.keys()

    def test_save_velocity_components(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_velocity_components': True})
        _delta = DeltaModel(input_file=p)
        assert _delta._save_any_grids is True
        assert len(_delta._save_fig_list) == 0
        assert _delta._save_velocity_components is True
        assert 'velocity_x' in _delta._save_var_list.keys()
        assert 'velocity_y' in _delta._save_var_list.keys()

    def test_save_eta_figs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_eta_figs': True})
        _delta = DeltaModel(input_file=p)
        assert len(_delta._save_fig_list) > 0
        assert _delta._save_any_grids is False

    def test_save_depth_figs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_depth_figs': True})
        _delta = DeltaModel(input_file=p)
        assert len(_delta._save_fig_list) > 0
        assert _delta._save_any_grids is False

    def test_save_stage_figs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_stage_figs': True})
        _delta = DeltaModel(input_file=p)
        assert len(_delta._save_fig_list) > 0
        assert _delta._save_any_grids is False

    def test_save_discharge_figs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_discharge_figs': True})
        _delta = DeltaModel(input_file=p)
        assert len(_delta._save_fig_list) > 0
        assert _delta._save_any_grids is False

    def test_save_velocity_figs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_velocity_figs': True})
        _delta = DeltaModel(input_file=p)
        assert len(_delta._save_fig_list) > 0
        assert _delta._save_any_grids is False

    def test_save_sedflux_figs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_sedflux_figs': True})
        _delta = DeltaModel(input_file=p)
        assert len(_delta._save_fig_list) > 0
        assert _delta._save_any_grids is False

    def test_save_sandfrac_figs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_sandfrac_figs': True})
        _delta = DeltaModel(input_file=p)
        assert len(_delta._save_fig_list) > 0
        assert _delta._save_any_grids is False

    def test_save_figs_sequential(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_figs_sequential': False})
        _delta = DeltaModel(input_file=p)
        assert len(_delta._save_fig_list) == 0
        assert _delta._save_any_grids is False
        assert _delta._save_figs_sequential is False

    def test_toggle_subsidence(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'toggle_subsidence': True})
        _delta = DeltaModel(input_file=p)
        assert _delta.toggle_subsidence is True

    def test_start_subsidence(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'start_subsidence': 12345})
        _delta = DeltaModel(input_file=p)
        assert _delta.start_subsidence == 12345

    def test_subsidence_rate(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'toggle_subsidence': True,
                                      'subsidence_rate': 1e-9})
        _delta = DeltaModel(input_file=p)
        assert _delta.subsidence_rate == 1e-9
        assert np.any(_delta.sigma > 0)
        assert np.all(_delta.sigma <= (1e-9 * _delta.dt))

    def test_sand_frac_bc(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'sand_frac_bc': -1})
        _delta = DeltaModel(input_file=p)
        assert _delta.sand_frac_bc == -1


class TestSettingOtherParametersFromYAMLSettings:

    def test_theta_sand(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'coeff_theta_sand': 1.4,
                                      'theta_water': 1.2})
        _delta = DeltaModel(input_file=p)
        assert _delta.theta_sand == 1.68

    def test_theta_mud(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'coeff_theta_mud': 0.8,
                                      'theta_water': 1.3})
        _delta = DeltaModel(input_file=p)
        assert _delta.theta_mud == 1.04

    def test_U_dep_mud(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'coeff_U_dep_mud': 0.4325,
                                      'u0': 2.2})
        _delta = DeltaModel(input_file=p)
        assert _delta.U_dep_mud == 0.9515

    def test_U_ero_sand(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'coeff_U_ero_sand': 1.23,
                                      'u0': 2.2})
        _delta = DeltaModel(input_file=p)
        assert _delta.U_ero_sand == 2.706

    def test_U_ero_mud(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'coeff_U_ero_mud': 1.67,
                                      'u0': 2.2})
        _delta = DeltaModel(input_file=p)
        assert _delta.U_ero_mud == 3.674

    def test_L0(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'L0_meters': 100,
                                      'Length': 6000,
                                      'dx': 5})
        _delta = DeltaModel(input_file=p)
        assert _delta.L0 == 20

    def test_N0(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5})
        _delta = DeltaModel(input_file=p)
        assert _delta.N0 == 100

    def test_L(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'Length': 1600,
                                      'dx': 20})
        _delta = DeltaModel(input_file=p)
        assert _delta.L == 80

    def test_W(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'Width': 1200,
                                      'dx': 20})
        _delta = DeltaModel(input_file=p)
        assert _delta.W == 60

    def test_u_max(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'u0': 2.3})
        _delta = DeltaModel(input_file=p)
        assert _delta.u_max == 4.6   # == 2*u0

    def test_C0(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'C0_percent': 10})
        _delta = DeltaModel(input_file=p)
        assert _delta.C0 == 0.1

    def test_dry_depth(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'h0': 0.5})
        _delta = DeltaModel(input_file=p)
        assert _delta.dry_depth == 0.05

    def test_dry_depth_limiter(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'h0': 20})
        _delta = DeltaModel(input_file=p)
        assert _delta.dry_depth == 0.1

    def test_CTR(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'Length': 4000,
                                      'Width': 6000,
                                      'dx': 10})
        _delta = DeltaModel(input_file=p)
        assert _delta.CTR == 299  # 300th index

    def test_CTR_small(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'Length': 40,
                                      'Width': 40,
                                      'dx': 10})
        _delta = DeltaModel(input_file=p)
        # small case, instead of 4/2-1=1 it is 4/2=2
        assert _delta.CTR == 2

    def test_gamma(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'S0': 0.01,
                                      'dx': 10,
                                      'u0': 3})
        _delta = DeltaModel(input_file=p)
        assert _delta.gamma == pytest.approx(0.10900000)

    def test_V0(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'h0': 3,
                                      'dx': 15})
        _delta = DeltaModel(input_file=p)
        assert _delta.V0 == 675

    def test_Qw0(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'u0': 0.8,
                                      'h0': 2,
                                      'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5})
        _delta = DeltaModel(input_file=p)
        assert _delta.N0 == 100
        assert _delta.Qw0 == 800

    def test_qw0(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'u0': 0.8,
                                      'h0': 3})
        _delta = DeltaModel(input_file=p)
        assert _delta.qw0 == pytest.approx(2.4)

    def test_Qp_water(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'u0': 0.8,
                                      'h0': 2,
                                      'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5,
                                      'Np_water': 2300})
        _delta = DeltaModel(input_file=p)
        assert _delta.N0 == 100
        assert _delta.Qw0 == 800
        assert _delta.Qp_water == pytest.approx(0.347826087)

    def test_dVs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'u0': 0.8,
                                      'h0': 2,
                                      'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5})
        _delta = DeltaModel(input_file=p)
        assert _delta.V0 == 50
        assert _delta.N0 == 100
        assert _delta.Qw0 == 800
        assert _delta.dVs == 50000

    def test_Qs0(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'C0_percent': 10,
                                      'u0': 0.8,
                                      'h0': 2,
                                      'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5})
        _delta = DeltaModel(input_file=p)
        assert _delta.Qw0 == 800
        assert _delta.Qs0 == pytest.approx(800 * 0.1)

    def test_Vp_sed(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'u0': 0.8,
                                      'h0': 2,
                                      'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5,
                                      'Np_sed': 1450})
        _delta = DeltaModel(input_file=p)
        assert _delta.dVs == 50000
        assert _delta.Vp_sed == 50000 / 1450

    def test_stepmax_and_size_indices(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'Length': 1600,
                                      'Width': 1200,
                                      'dx': 20})
        _delta = DeltaModel(input_file=p)
        assert _delta.L == 80
        assert _delta.W == 60
        assert _delta.stepmax == (80 + 60) * 2
        assert _delta.size_indices == (80 + 60)

    def test_dt(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'C0_percent': 10,
                                      'u0': 0.8,
                                      'h0': 2,
                                      'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5})
        _delta = DeltaModel(input_file=p)
        assert _delta.Qw0 == 800
        assert _delta.dVs == 50000
        assert _delta.Qs0 == pytest.approx(800 * 0.1)
        assert _delta.dt == 625

    def test_omega_flow_iter(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'itermax': 7})
        _delta = DeltaModel(input_file=p)
        assert _delta.omega_flow_iter == pytest.approx(2 / 7)

    def test_N_crossdiff(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'u0': 0.8,
                                      'h0': 2,
                                      'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5})
        _delta = DeltaModel(input_file=p)
        assert _delta.V0 == 50
        assert _delta.N0 == 100
        assert _delta.Qw0 == 800
        assert _delta.dVs == 50000
        assert _delta.N_crossdiff == int(round(50000 / 50))

    def test_diffusion_multiplier(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'u0': 0.8,
                                      'h0': 2,
                                      'N0_meters': 500,
                                      'Width': 6000,
                                      'dx': 5,
                                      'alpha': 0.3,
                                      'C0_percent': 10})
        _delta = DeltaModel(input_file=p)
        assert _delta.V0 == 50
        assert _delta.N0 == 100
        assert _delta.Qw0 == 800
        assert _delta.dVs == 50000
        assert _delta.Qs0 == pytest.approx(800 * 0.1)
        assert _delta.dt == 625
        assert _delta.N_crossdiff == int(round(50000 / 50))
        assert _delta.diffusion_multiplier == (625 / 1000 * 0.3 * 0.5 / 5**2)

    def test_active_layer_thickness_float(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'active_layer_thickness': 2.7})
        _delta = DeltaModel(input_file=p)
        assert _delta.active_layer_thickness == 2.7

    def test_active_layer_thickness_int(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'active_layer_thickness': 2})
        _delta = DeltaModel(input_file=p)
        assert _delta.active_layer_thickness == 2

    def test_active_layer_thickness_default(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)
        assert _delta.active_layer_thickness == _delta.h0 / 2


class TestInitMetadataList:

    def test_save_list_exists(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        f.close()
        delta = DeltaModel(input_file=p)
        # check things about the metadata
        assert hasattr(delta, '_save_var_list')
        assert type(delta._save_var_list) == dict
        assert 'meta' in delta._save_var_list.keys()
        # save meta not on, so check that it is empty
        assert delta._save_var_list['meta'] == {}

    def test_default_meta_list(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'save_metadata', True)
        utilities.write_parameter_to_file(f, 'legacy_netcdf', False)
        f.close()
        delta = DeltaModel(input_file=p)
        # check things about the metadata
        assert hasattr(delta, '_save_var_list')
        assert type(delta._save_var_list) == dict
        assert 'meta' in delta._save_var_list.keys()
        # save meta on, so check that some expected values are there
        assert 'L0' in delta._save_var_list['meta'].keys()
        assert delta._save_var_list['meta']['L0'] == ['L0', 'cells', 'i8', ()]
        assert 'H_SL' in delta._save_var_list['meta'].keys()
        assert delta._save_var_list['meta']['H_SL'] == \
            [None, 'meters', 'f4', 'time']

    def test_default_meta_list_legacy(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'save_metadata', True)
        utilities.write_parameter_to_file(f, 'legacy_netcdf', True)
        f.close()
        delta = DeltaModel(input_file=p)
        # check things about the metadata
        assert hasattr(delta, '_save_var_list')
        assert type(delta._save_var_list) == dict
        assert 'meta' in delta._save_var_list.keys()
        # save meta on, so check that some expected values are there
        assert 'L0' in delta._save_var_list['meta'].keys()
        assert delta._save_var_list['meta']['L0'] == ['L0', 'cells', 'i8', ()]
        assert 'H_SL' in delta._save_var_list['meta'].keys()
        assert delta._save_var_list['meta']['H_SL'] == \
            [None, 'meters', 'f4', 'total_time']

    def test_netcdf_vars(self, tmp_path):
        # test that stuff makes it to the netcdf file as expected
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        utilities.write_parameter_to_file(f, 'save_metadata', True)
        utilities.write_parameter_to_file(f, 'legacy_netcdf', False)
        f.close()
        delta = DeltaModel(input_file=p)
        # check things about the metadata
        assert hasattr(delta, '_save_var_list')
        assert type(delta._save_var_list) == dict
        assert 'meta' in delta._save_var_list.keys()
        # save meta on, so check that some expected values are there
        assert 'L0' in delta._save_var_list['meta'].keys()
        assert delta._save_var_list['meta']['L0'] == ['L0', 'cells', 'i8', ()]
        assert 'H_SL' in delta._save_var_list['meta'].keys()
        assert delta._save_var_list['meta']['H_SL'] == \
            [None, 'meters', 'f4', 'time']
        # check save var list for eta
        assert 'eta' in delta._save_var_list.keys()
        assert delta._save_var_list['eta'] == \
            ['eta', 'meters', 'f4', ('time', 'x', 'y')]
        # force save to netcdf
        delta.save_grids_and_figs()
        # close netcdf
        delta.output_netcdf.close()
        # check out the netcdf
        data = Dataset(os.path.join(delta.prefix, 'pyDeltaRCM_output.nc'),
                       'r+', format='NETCDF4')
        # check for meta group
        assert 'meta' in data.groups
        # check for L0 a single value metadata
        assert 'L0' in data['meta'].variables
        assert data['meta']['L0'][0].data == delta.L0
        # check H_SL a vector of metadata
        assert 'H_SL' in data['meta'].variables
        assert data['meta']['H_SL'].dimensions == ('time',)
        assert data['time'].shape == data['meta']['H_SL'].shape
        # check on the eta grid
        assert 'eta' in data.variables
        assert data['eta'].shape[0] == data['time'].shape[0]
        assert data['eta'].shape[1] == delta.L
        assert data['eta'].shape[2] == delta.W

    def test_netcdf_vars_legacy(self, tmp_path):
        # test that stuff makes it to the netcdf file as expected
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        utilities.write_parameter_to_file(f, 'save_metadata', True)
        utilities.write_parameter_to_file(f, 'legacy_netcdf', True)
        f.close()
        delta = DeltaModel(input_file=p)
        # check things about the metadata
        assert hasattr(delta, '_save_var_list')
        assert type(delta._save_var_list) == dict
        assert 'meta' in delta._save_var_list.keys()
        # save meta on, so check that some expected values are there
        assert 'L0' in delta._save_var_list['meta'].keys()
        assert delta._save_var_list['meta']['L0'] == ['L0', 'cells', 'i8', ()]
        assert 'H_SL' in delta._save_var_list['meta'].keys()
        assert delta._save_var_list['meta']['H_SL'] == \
            [None, 'meters', 'f4', 'total_time']
        # check save var list for eta
        assert 'eta' in delta._save_var_list.keys()
        assert delta._save_var_list['eta'] == \
            ['eta', 'meters', 'f4', ('total_time', 'length', 'width')]
        # force save to netcdf
        delta.save_grids_and_figs()
        # close netcdf
        delta.output_netcdf.close()
        # check out the netcdf
        data = Dataset(os.path.join(delta.prefix, 'pyDeltaRCM_output.nc'),
                       'r+', format='NETCDF4')
        # check for meta group
        assert 'meta' in data.groups
        # check for L0 a single value metadata
        assert 'L0' in data['meta'].variables
        assert data['meta']['L0'][0].data == delta.L0
        # check H_SL a vector of metadata
        assert 'H_SL' in data['meta'].variables
        assert data['meta']['H_SL'].dimensions == ('total_time',)
        assert data['time'].shape == data['meta']['H_SL'].shape
        # check on the eta grid
        assert 'eta' in data.variables
        assert data['eta'].shape[0] == data['time'].shape[0]
        assert data['eta'].shape[1] == delta.L
        assert data['eta'].shape[2] == delta.W
