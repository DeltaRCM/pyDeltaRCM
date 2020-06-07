# unit tests for deltaRCM_tools.py

import pytest

import sys
import os
import numpy as np

import glob

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
    assert not 'Timestep: 0.0' in captd.out


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
    assert captd1.out == ''
    assert len(glob.glob(os.path.join(delta.prefix, '*.log'))
               ) == 1  # log file exists
    assert 'Timestep: 0.0' in captd2.out  # if verbose >= 1
    assert not 'Creating output directory' in captd2.out  # goes to logger


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
    assert 'Timestep: 0.0' in captd2.out  # if verbose >= 1
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
    _logs = glob.glob(os.path.join(delta.prefix, '*.log'))
    assert len(_logs) == 1  # log file exists
    for _ in range(0, 2):
        delta.update()
    assert len(_logs) == 1  # log file exists, still only one
    with open(_logs[0], 'r') as _logfile:
        _lines = _logfile.readlines()
        _lines = ' '.join(_lines)  # collapse to a single string
        assert '---- Timestep 0.0 ----' in _lines
        assert '---- Timestep 1.0 ----' in _lines
        assert not '---- Timestep 2.0 ----' in _lines


@pytest.mark.xfail(raises=ValueError, strict=True)
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
