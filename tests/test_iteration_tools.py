# unit tests for deltaRCM_tools.py

import pytest

import sys
import os
import numpy as np

import glob
import netCDF4

from pyDeltaRCM.model import DeltaModel

from utilities import test_DeltaModel
import utilities


def test_run_one_timestep(test_DeltaModel):
    test_DeltaModel.run_one_timestep()
    # basically assume sediment has been added at inlet
    assert test_DeltaModel.H_SL == 0.0
    assert test_DeltaModel.qs[0, 4] != 0.


def test_finalize_timestep(test_DeltaModel):
    test_DeltaModel.finalize_timestep()
    # check that sea level rose as expected
    assert test_DeltaModel.H_SL == 0.3


def test_subsidence_in_update(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'toggle_subsidence': True,
                                  'sigma_max': 1e-8,
                                  'start_subsidence': 0,
                                  'seed': 0})
    _delta = DeltaModel(input_file=p)
    assert _delta.dt == 20000
    assert _delta.sigma_max == 1e-8
    assert _delta.sigma[17, 5] == 0.0  # outside the sigma mask
    assert _delta.sigma[17, 6] == 0.0002  # inside the sigma mask
    assert np.all(_delta.eta[17, 5:7] == -_delta.h0)
    _delta.update()
    assert _delta.eta[17, 5] == pytest.approx(-_delta.h0)
    assert _delta.eta[17, 6] == pytest.approx(-_delta.h0 - 0.0002)
    _delta.output_netcdf.close()


def test_subsidence_in_update_delayed_start(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'toggle_subsidence': True,
                                  'sigma_max': 1e-8,
                                  'start_subsidence': 20000,
                                  'seed': 0})
    _delta = DeltaModel(input_file=p)
    assert _delta.dt == 20000
    assert _delta.sigma_max == 1e-8
    assert _delta.sigma[17, 5] == 0.0  # outside the sigma mask
    assert _delta.sigma[17, 6] == 0.0002  # inside the sigma mask
    assert np.all(_delta.eta[17, 5:7] == -_delta.h0)
    _delta.update()  # no subsidence applied
    assert _delta.time == 20000
    assert _delta.eta[17, 5] == pytest.approx(-_delta.h0)
    assert _delta.eta[17, 6] == pytest.approx(-_delta.h0)
    _delta.update()
    assert _delta.time == 40000
    assert _delta.eta[17, 5] == pytest.approx(-_delta.h0)
    assert _delta.eta[17, 6] == pytest.approx(-_delta.h0 - 0.0002)
    _delta.output_netcdf.close()


def test_subsidence_changed_with_timestep(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'toggle_subsidence': True,
                                  'sigma_max': 1e-8})
    _delta = DeltaModel(input_file=p)
    assert _delta.dt == 20000
    assert _delta.sigma[17, 6] == 0.0002
    _delta.time_step = 86400
    assert _delta.sigma[17, 6] == 0.000864
    _delta.output_netcdf.close()


def test_expand_stratigraphy(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'verbose', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    utilities.write_parameter_to_file(f, 'save_dt', 600)
    utilities.write_parameter_to_file(f, 'toggle_subsidence', True)
    utilities.write_parameter_to_file(f, 'sigma_max', 1e-8)
    utilities.write_parameter_to_file(f, 'start_subsidence', 20000)
    utilities.write_parameter_to_file(f, 'seed', 0)
    f.close()
    _delta = DeltaModel(input_file=p)
    assert _delta.dt == 300
    assert _delta.n_steps == 10
    assert _delta.strata_counter == 0
    assert _delta.strata_eta[:, _delta.strata_counter].getnnz() == 0
    for _t in range(19):
        assert _delta.strata_eta[:, _delta.strata_counter].getnnz() == 0
        _delta.update()
        assert _delta.time == _delta.dt * (_t + 1)
        assert _delta.strata_eta.shape[1] == 10
    assert _delta.time == 19 * 300
    assert _delta.strata_counter == 10  # stored 10 but invalid index next store
    assert _delta.strata_eta.shape[1] == 10
    # nothing occurs on next  update, because save_dt = 2 * dt
    _delta.update()
    assert _delta.time == 20 * 300
    assert _delta.strata_counter == 10
    assert _delta.strata_eta.shape[1] == 10
    # expansion occurs when model tries to save strata after next update
    _delta.update()
    assert _delta.time == 21 * 300
    assert _delta.strata_counter == 11
    assert _delta.strata_eta.shape[1] == 20
    # run to bring to even 100 steps, check status again
    for _t in range(79):
        _delta.update()
    assert _delta.time == 100 * 300
    assert _delta.strata_counter == 50
    assert _delta.strata_eta.shape[1] == 50


def test_verbose_printing_0(tmp_path, capsys):
    """
    This test should create the log, and then print nothing at all.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'verbose', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    f.close()
    delta = DeltaModel(input_file=p)
    assert os.path.isfile(os.path.join(delta.prefix, 'pyDeltaRCM_output.nc'))
    assert len(glob.glob(os.path.join(delta.prefix, '*.log'))
               ) == 1  # log file exists
    delta.update()
    captd = capsys.readouterr()
    assert 'Model time: 0.0' not in captd.out


def test_verbose_printing_1(tmp_path, capsys):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    f.close()
    delta = DeltaModel(input_file=p)
    captd1 = capsys.readouterr()
    delta.update()
    captd2 = capsys.readouterr()
    log_glob = glob.glob(os.path.join(delta.prefix, '*.log'))
    assert len(log_glob) == 1  # log file exists
    assert 'Model time: 0.0' in captd1.out  # if verbose >= 1
    assert 'Model time: 300.0' in captd2.out  # if verbose >= 1
    assert 'Creating output directory' not in captd2.out  # goes to logger


def test_verbose_printing_2(tmp_path, capsys):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'verbose', 2)
    utilities.write_parameter_to_file(f, 'seed', 10)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    f.close()
    delta = DeltaModel(input_file=p)
    captd1 = capsys.readouterr()
    delta.update()
    captd2 = capsys.readouterr()
    assert len(glob.glob(os.path.join(delta.prefix, '*.log'))
               ) == 1  # log file exists
    assert 'Setting random seed to' in captd1.out   # if verbose >= 2
    assert 'Model time: 0.0' in captd1.out  # if verbose >= 1
    assert 'Model time: 300.0' in captd2.out  # if verbose >= 1
    assert delta.seed == 10


def test_logger_has_initialization_lines(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'seed', 10)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    f.close()
    delta = DeltaModel(input_file=p)
    _logs = glob.glob(os.path.join(delta.prefix, '*.log'))
    assert len(_logs) == 1  # log file exists
    with open(_logs[0], 'r') as _logfile:
        _lines = _logfile.readlines()
        _lines = ' '.join(_lines)  # collapse to a single string
        assert 'Setting model constant' in _lines
        assert 'Setting random seed to: 10' in _lines
        assert 'Random seed is: 10' in _lines
        assert 'Creating model domain' in _lines
        assert 'Generating netCDF file for output grids' in _lines
        assert 'Output netCDF file created' in _lines
        assert 'Model initialization complete' in _lines
    assert not os.path.isfile(os.path.join(
        tmp_path, 'out_dir', 'discharge_0.0.png'))
    assert not os.path.isfile(os.path.join(tmp_path, 'out_dir', 'eta_0.0.png'))


def test_logger_has_timestep_lines(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'seed', 10)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.time_step == 300
    _logs = glob.glob(os.path.join(delta.prefix, '*.log'))
    assert len(_logs) == 1  # log file exists
    for _ in range(0, 2):
        delta.update()
    assert len(_logs) == 1  # log file exists, still only one
    with open(_logs[0], 'r') as _logfile:
        _lines = _logfile.readlines()
        _lines = ' '.join(_lines)  # collapse to a single string
        assert '---- Model time 0.0 ----' in _lines
        assert '---- Model time 300.0 ----' in _lines
        assert '---- Model time 600.0 ----' in _lines
        assert '---- Model time 900.0 ----' not in _lines


def test_logger_random_seed_always_recorded(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'verbose', 0)
    # do not set the seed explicitly, let it be set by the model
    # utilities.write_parameter_to_file(f, 'seed', None)
    f.close()
    delta = DeltaModel(input_file=p)
    _logs = glob.glob(os.path.join(delta.prefix, '*.log'))
    assert len(_logs) == 1  # log file exists
    with open(_logs[0], 'r') as _logfile:
        _lines = _logfile.readlines()
        _joinedlines = ' '.join(_lines)  # collapse to a single string
        assert 'Random seed is: ' in _joinedlines

        # determine the index of the line
        _idx = ['Random seed is: ' in _l for _l in _lines]
        assert sum(_idx) == 1  # one and only one True in list
        _idx = _idx.index(True)

        # try to covert to int, otherwise fail
        _seed = _lines[_idx].split(':')[-1]  # pull the seed value
        try:
            _intseed = int(_seed)
        except ValueError:
            raise ValueError('Could not convert the seed to int')

        assert _intseed >= 0


def test_save_no_figs_no_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    utilities.write_parameter_to_file(f, 'save_dt', 5)
    utilities.write_parameter_to_file(f, 'save_strata', False)
    f.close()

    _delta = DeltaModel(input_file=p)
    img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
    nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
    assert len(img_glob) == 0
    assert len(nc_glob) == 0

    for _t in range(0, 4):
        _delta.update()
    assert _delta.time_iter == 4.0
    img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
    nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
    assert len(img_glob) == 0
    assert len(nc_glob) == 0

    _delta.update()
    assert _delta.time_iter == 5.0
    img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
    nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
    assert len(img_glob) == 0
    assert len(nc_glob) == 0


def test_save_one_fig_no_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'save_strata', False)
    f.close()

    _delta = DeltaModel(input_file=p)
    img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
    nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
    assert len(img_glob) == 0
    assert len(nc_glob) == 0

    for _ in range(0, 2):
        _delta.update()
    assert _delta.time_iter == 2.0

    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    exp_path_png0 = os.path.join(tmp_path / 'out_dir', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'out_dir', 'eta_00001.png')
    exp_path_png2 = os.path.join(tmp_path / 'out_dir', 'eta_00002.png')
    assert not os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png0)
    assert os.path.isfile(exp_path_png1)
    assert not os.path.isfile(exp_path_png2)


def test_save_one_fig_one_grid(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    utilities.write_parameter_to_file(f, 'save_discharge_figs', True)
    utilities.write_parameter_to_file(f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'save_strata', True)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)
    nc_size_before = os.path.getsize(exp_path_nc)
    assert nc_size_before > 0

    # update a couple times, should not increase size until finalize()
    for _ in range(0, 2):
        _delta.update()
    nc_size_middle = os.path.getsize(exp_path_nc)
    assert _delta.time_iter == 2.0
    assert nc_size_middle > nc_size_before

    # now finalize, and then file size should increase
    _delta.finalize()
    nc_size_after = os.path.getsize(exp_path_nc)
    assert _delta.time_iter == 2.0
    assert nc_size_after > nc_size_middle
    assert nc_size_after > nc_size_before


def test_save_all_figures_no_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    utilities.write_parameter_to_file(f, 'save_discharge_figs', True)
    utilities.write_parameter_to_file(f, 'save_velocity_figs', True)
    utilities.write_parameter_to_file(f, 'save_stage_figs', True)
    utilities.write_parameter_to_file(f, 'save_depth_figs', True)
    utilities.write_parameter_to_file(f, 'save_sedflux_figs', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'save_strata', False)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert not os.path.isfile(exp_path_nc)

    for _ in range(0, 2):
        _delta.update()

    exp_path_png0 = os.path.join(tmp_path / 'out_dir', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'out_dir', 'depth_00000.png')
    exp_path_png2 = os.path.join(tmp_path / 'out_dir', 'stage_00000.png')
    exp_path_png3 = os.path.join(tmp_path / 'out_dir', 'velocity_00000.png')
    exp_path_png4 = os.path.join(tmp_path / 'out_dir', 'discharge_00000.png')
    exp_path_png5 = os.path.join(tmp_path / 'out_dir', 'sedflux_00000.png')
    assert os.path.isfile(exp_path_png0)
    assert os.path.isfile(exp_path_png1)
    assert os.path.isfile(exp_path_png2)
    assert os.path.isfile(exp_path_png3)
    assert os.path.isfile(exp_path_png4)
    assert os.path.isfile(exp_path_png5)


def test_save_all_figures_sequential_false(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    utilities.write_parameter_to_file(f, 'save_discharge_figs', True)
    utilities.write_parameter_to_file(f, 'save_velocity_figs', True)
    utilities.write_parameter_to_file(f, 'save_stage_figs', True)
    utilities.write_parameter_to_file(f, 'save_depth_figs', True)
    utilities.write_parameter_to_file(f, 'save_sedflux_figs', True)
    utilities.write_parameter_to_file(f, 'save_figs_sequential', False)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'save_strata', False)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert not os.path.isfile(exp_path_nc)

    for _ in range(0, 5):
        _delta.update()
    exp_path_png0 = os.path.join(tmp_path / 'out_dir', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'out_dir', 'depth_00000.png')
    exp_path_png0_latest = os.path.join(tmp_path / 'out_dir', 'eta_latest.png')
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


def test_save_metadata_no_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_metadata', True)
    utilities.write_parameter_to_file(f, 'save_eta_grids', False)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    for _ in range(0, 2):
        _delta.update()
    assert _delta.time_iter == 2.0
    _delta.finalize()

    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    assert not ('eta' in ds.variables)
    assert ds['meta']['H_SL'].shape[0] == 3
    assert ds['meta']['L0'][:] == 1


def test_save_metadata_and_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_metadata', True)
    utilities.write_parameter_to_file(f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(f, 'save_velocity_grids', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.25)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    for _ in range(0, 2):
        _delta.update()
    assert _delta.time_iter == 2.0
    _delta.finalize()

    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    assert ('eta' in ds.variables)
    assert ('velocity' in ds.variables)
    assert ds['meta']['H_SL'].shape[0] == 3
    assert ds['meta']['L0'][:] == 1
    assert np.all(ds['meta']['f_bedload'][:] == 0.25)


def test_save_one_grid_metadata_by_default(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(f, 'save_metadata', False)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'C0_percent', 0.2)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    for _ in range(0, 6):
        _delta.update()
    assert _delta.time_iter == 6.0
    _delta.finalize()

    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    _arr = ds.variables['eta']
    assert _arr.shape[1] == _delta.eta.shape[0]
    assert _arr.shape[2] == _delta.eta.shape[1]
    assert ('meta' in ds.groups)  # if any grids, save meta too
    assert ds.groups['meta']['H_SL'].shape[0] == _arr.shape[0]
    assert np.all(ds.groups['meta']['C0_percent'][:] == 0.2)
    assert np.all(ds.groups['meta']['f_bedload'][:] == 0.5)


def test_save_eta_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    for _ in range(0, 2):
        _delta.update()
    assert _delta.time_iter == 2.0
    _delta.finalize()

    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    _arr = ds.variables['eta']
    assert _arr.shape[1] == _delta.eta.shape[0]
    assert _arr.shape[2] == _delta.eta.shape[1]
    assert ('meta' in ds.groups)  # if any grids, save meta too


def test_save_depth_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    for _ in range(0, 2):
        _delta.update()
    assert _delta.time_iter == 2.0
    _delta.finalize()

    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    _arr = ds.variables['depth']
    assert _arr.shape[1] == _delta.depth.shape[0]
    assert _arr.shape[2] == _delta.depth.shape[1]
    assert ('meta' in ds.groups)  # if any grids, save meta too


def test_save_velocity_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_velocity_grids', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
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


def test_save_stage_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_stage_grids', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    for _ in range(0, 2):
        _delta.update()
    assert _delta.time_iter == 2.0
    _delta.finalize()

    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    _arr = ds.variables['stage']
    assert _arr.shape[1] == _delta.eta.shape[0]
    assert _arr.shape[2] == _delta.eta.shape[1]
    assert ('meta' in ds.groups)  # if any grids, save meta too


def test_save_discharge_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    for _ in range(0, 2):
        _delta.update()
    assert _delta.time_iter == 2.0
    _delta.finalize()

    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    _arr = ds.variables['discharge']
    assert _arr.shape[1] == _delta.eta.shape[0]
    assert _arr.shape[2] == _delta.eta.shape[1]
    assert ('meta' in ds.groups)  # if any grids, save meta too


def test_save_sedflux_grids(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'itermax', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'save_sedflux_grids', True)
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    f.close()

    _delta = DeltaModel(input_file=p)
    exp_path_nc = os.path.join(tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    for _ in range(0, 2):
        _delta.update()
    assert _delta.time_iter == 2.0
    _delta.finalize()

    ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
    _arr = ds.variables['sedflux']
    assert _arr.shape[1] == _delta.eta.shape[0]
    assert _arr.shape[2] == _delta.eta.shape[1]
    assert ('meta' in ds.groups)  # if any grids, save meta too
