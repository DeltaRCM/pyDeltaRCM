
import pytest

import os
import shutil
import locale
import numpy as np
import subprocess
import glob
import netCDF4
import time
import platform

import pyDeltaRCM as _pyimportedalias
from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM import shared_tools
from pyDeltaRCM import preprocessor

import utilities
from utilities import test_DeltaModel


# test yaml parsing
def test_override_from_testfile(test_DeltaModel):
    out_path = test_DeltaModel.out_dir.split(os.path.sep)
    assert out_path[-1] == 'out_dir'
    assert test_DeltaModel.Length == 10


def test_error_if_no_file_found():
    with pytest.raises(FileNotFoundError):
        delta = DeltaModel(input_file='./nonexisting_file.yaml')


def test_override_single_default(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'S0', 0.005)
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.S0 == 0.005


def test_override_two_defaults(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'S0', 0.005)
    utilities.write_parameter_to_file(f, 'Np_sed', 2)
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.S0 == 0.005
    assert delta.Np_sed == 2


def test_override_bad_type_float_string(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'S0', 'a string?!')
    f.close()
    with pytest.raises(TypeError):
        delta = DeltaModel(input_file=p)


def test_override_bad_type_int_float(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'beta', 24.4234)
    f.close()
    with pytest.raises(TypeError):
        delta = DeltaModel(input_file=p)


def test_not_creating_illegal_attributes(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'illegal_attribute', True)
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.S0 == 0.0002  # from default.yaml
    assert not hasattr(delta, 'illegal_attribute')


def test_not_overwriting_existing_attributes(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'input_file', '/fake/path.yaml')
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.S0 == 0.0002  # from default.yaml
    assert hasattr(delta, 'input_file')
    assert delta.input_file == p


# paramter specific yaml settings

def test_random_seed_settings_value(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 9999)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    f.close()
    shared_tools.set_random_seed(9999)
    _preval_same = shared_tools.get_random_uniform(1)
    shared_tools.set_random_seed(5)
    _preval_diff = shared_tools.get_random_uniform(1000)
    delta = DeltaModel(input_file=p)
    assert delta.seed == 9999
    _postval_same = shared_tools.get_random_uniform(1)
    assert _preval_same == _postval_same
    assert delta.seed == 9999


def test_random_seed_settings_newinteger_default(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'S0', 0.005)
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.seed is not None
    assert delta.seed <= (2**32) - 1
    assert isinstance(int(delta.seed), int)


def test_no_outputs_save_dt_notreached(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'save_strata', True)
    utilities.write_parameter_to_file(f, 'save_dt', 43200)
    f.close()
    delta = DeltaModel(input_file=p)
    for _ in range(2):
        delta.update()
    assert delta.dt == 20000.0
    assert delta.strata_counter == 1  # one saved, t==0
    assert delta.time < delta.save_dt
    delta.finalize()


def test_no_outputs_save_strata_false(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'save_strata', False)
    utilities.write_parameter_to_file(f, 'save_dt', 21600)
    f.close()
    delta = DeltaModel(input_file=p)
    for _ in range(2):
        delta.update()
    assert delta.dt == 20000.0
    assert not hasattr(delta, 'strata_counter')
    assert delta.time > delta.save_dt


# test the entry points
def test_entry_point_installed_call(tmp_path):
    """
    test calling the command line feature with a config file.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'timesteps', 2)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    subprocess.check_output(['pyDeltaRCM',
                             '--config', str(p)])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png0 = os.path.join(tmp_path / 'test', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png0)
    assert os.path.isfile(exp_path_png1)


def test_entry_point_python_main_call(tmp_path):
    """
    test calling the python hook command line feature with a config file.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'timesteps', 1)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                             '--config', str(p)])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
    exp_path_png2 = os.path.join(tmp_path / 'test', 'eta_00002.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png)
    assert os.path.isfile(exp_path_png1)
    assert not os.path.isfile(exp_path_png2)


def test_entry_point_python_main_call_dryrun(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'timesteps', 1)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                             '--config', str(p),
                             '--dryrun'])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
    assert not os.path.isfile(exp_path_nc)   # does not exist because --dryrun
    assert not os.path.isfile(exp_path_png)  # does not exist because --dryrun


def test_entry_point_python_main_call_timesteps(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                             '--config', str(p),
                             '--timesteps', '2'])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png)


def test_error_if_no_timesteps(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                 '--config', str(p)])


def test_entry_point_timesteps(tmp_path):
    """
    test calling the command line feature with a config file.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    subprocess.check_output(['pyDeltaRCM',
                             '--config', str(p), '--timesteps', '2'])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png0 = os.path.join(tmp_path / 'test', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png0)
    assert os.path.isfile(exp_path_png1)


def test_entry_point_time(tmp_path):
    """
    test calling the command line feature with a config file.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    subprocess.check_output(['pyDeltaRCM',
                             '--config', str(p), '--time', '1000'])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png0 = os.path.join(tmp_path / 'test', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png0)
    assert os.path.isfile(exp_path_png1)


def test_version_call():
    """
    test calling the command line feature to query the version.
    """
    encoding = locale.getpreferredencoding()
    printed1 = subprocess.run(['pyDeltaRCM', '--version'],
                              stdout=subprocess.PIPE, encoding=encoding)
    _exp_str1 = 'pyDeltaRCM ' + _pyimportedalias.__version__ + '\n'
    assert printed1.stdout == _exp_str1
    printed2 = subprocess.run(
        ['python', '-m', 'pyDeltaRCM', '--version'],
        stdout=subprocess.PIPE, encoding=encoding)
    _exp_str2 = 'pyDeltaRCM ' + _pyimportedalias.__version__ + '\n'
    assert printed2.stdout == _exp_str2


# test high level python api
def test_py_hlvl_wo_args():
    with pytest.raises(ValueError):
        pp = preprocessor.Preprocessor()


def test_py_hlvl_wo_timesteps(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    pp = preprocessor.Preprocessor(p)
    with pytest.raises(ValueError,
                       match=r'You must specify a run duration *.'):
        pp.run_jobs()


def test_py_hlvl_tsteps_yml_runjobs_sngle(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'timesteps', 50)
    f.close()
    pp = preprocessor.Preprocessor(p)
    assert len(pp.file_list) == 1
    assert pp._is_completed is False
    pp.run_jobs()
    assert pp._is_completed is True
    end_time = utilities.read_endtime_from_log(tmp_path / 'test')
    assert end_time == 15000


def test_py_hlvl_time_yml_runjobs_sngle(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'time', 1000)
    f.close()
    pp = preprocessor.Preprocessor(p)
    assert len(pp.file_list) == 1
    pp.run_jobs()
    assert pp._is_completed is True
    end_time = utilities.read_endtime_from_log(tmp_path / 'test')
    assert end_time == 1200


def test_py_hlvl_time_If_yml_runjobs_sngle(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'time', 10000)
    utilities.write_parameter_to_file(f, 'If', 0.1)
    f.close()
    pp = preprocessor.Preprocessor(p)
    assert len(pp.file_list) == 1
    pp.run_jobs()
    assert pp._is_completed is True
    end_time = utilities.read_endtime_from_log(tmp_path / 'test')
    assert end_time == 1200


def test_py_hlvl_timeargs_precedence_tstepsovertime(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'time', 10000)
    utilities.write_parameter_to_file(f, 'timesteps', 2)
    f.close()
    pp = preprocessor.Preprocessor(p)
    pp.run_jobs()
    assert pp._is_completed is True
    assert pp.job_list[0].deltamodel.time == 600
    assert pp.job_list[0].deltamodel.time_iter == 2


def test_py_hlvl_timeargs_precedence_tstepsovertimeyears(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'time_years', 10000)
    utilities.write_parameter_to_file(f, 'timesteps', 2)
    f.close()
    pp = preprocessor.Preprocessor(p)
    pp.run_jobs()
    assert pp._is_completed is True
    assert pp.job_list[0].deltamodel.time == 600
    assert pp.job_list[0].deltamodel.time_iter == 2


def test_py_hlvl_timeargs_precedence_timeovertimeyears(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'time', 900)
    utilities.write_parameter_to_file(f, 'time_years', 10000)
    f.close()
    pp = preprocessor.Preprocessor(p)
    pp.run_jobs()
    assert pp._is_completed is True
    assert pp.job_list[0].deltamodel.time == 900
    assert pp.job_list[0].deltamodel.time_iter == 3


def test_py_hlvl_time_timestep_mismatch_endtime_check(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'time', 1000)
    f.close()
    pp = preprocessor.Preprocessor(p)
    pp.run_jobs()
    # timestep will have been 300, so should have run 4 iterations, and time
    # will equal 1200. I.e., more than specified, this is the expected
    # behavior
    assert pp._is_completed is True
    assert pp.job_list[0].deltamodel.time == 1200
    assert pp.job_list[0].deltamodel.time_iter == 4


def test_py_hlvl_timeyears_yml_runjobs_sngle(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'time_years', 3.16880878140e-05)
    f.close()
    pp = preprocessor.Preprocessor(p)
    assert len(pp.file_list) == 1
    pp.run_jobs()
    assert pp._is_completed is True
    end_time = utilities.read_endtime_from_log(tmp_path / 'test')
    assert end_time == 1200


def test_py_hlvl_timeyears_If_yml_runjobs_sngle(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'time_years', 0.00031688087814)
    utilities.write_parameter_to_file(f, 'If', 0.1)
    f.close()
    pp = preprocessor.Preprocessor(p)
    assert len(pp.file_list) == 1
    assert pp._is_completed is False
    pp.run_jobs()
    assert pp._is_completed is True
    end_time = utilities.read_endtime_from_log(tmp_path / 'test')
    assert end_time == 1200


def test_py_hlvl_args(tmp_path):
    """
    test calling the python hook command line feature with a config file.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 1
    assert pp._is_completed is False
    pp.run_jobs()
    assert type(pp.job_list[0].deltamodel) is DeltaModel
    assert pp.job_list[0].deltamodel.Length == 10.0
    assert pp.job_list[0].deltamodel.Width == 10.0
    assert pp.job_list[0].deltamodel.dx == 1.0
    assert pp.job_list[0].deltamodel.seed == 0
    assert issubclass(type(pp.job_list[0]), preprocessor._BaseJob)
    assert len(pp.file_list) == 1
    assert len(pp.job_list) == 1
    assert pp._is_completed is True
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
    exp_path_png2 = os.path.join(tmp_path / 'test', 'eta_00002.png')
    exp_path_png3 = os.path.join(tmp_path / 'test', 'eta_00003.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png)
    assert os.path.isfile(exp_path_png1)
    assert not os.path.isfile(exp_path_png3)


def test_py_hlvl_mtrx_1list_timesteps_arg(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'save_dt', 600)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload'],
                                   [[0.2, 0.6]])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    assert pp._has_matrix is True
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    pp.run_jobs()
    assert len(pp.file_list) == 2
    assert pp._is_completed is True
    f_bedload_list = [j.deltamodel.f_bedload for j in pp.job_list]
    assert sum([j == 0.2 for j in f_bedload_list]) == 1
    assert sum([j == 0.6 for j in f_bedload_list]) == 1
    end_time_000 = utilities.read_endtime_from_log(
        tmp_path / 'test' / 'job_000')
    end_time_001 = utilities.read_endtime_from_log(
        tmp_path / 'test' / 'job_000')
    assert end_time_000 == 900.0
    assert end_time_001 == 900.0
    assert pp.job_list[1].deltamodel.dt == 300.0
    assert pp._is_completed is True
    end_time = utilities.read_endtime_from_log(tmp_path / 'test' / 'job_000')
    assert end_time == pp.job_list[1].deltamodel.dt * 3
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc1 = os.path.join(
        tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc1)
    _logs = glob.glob(os.path.join(pp.job_list[0].deltamodel.prefix, '*.log'))
    assert len(_logs) == 1  # log file exists
    with open(_logs[0], 'r') as _logfile:
        _lines = _logfile.readlines()
        _lines = ' '.join(_lines)  # collapse to a single string
        assert 'Time: 0.0' in _lines
        assert 'Time: 300.0' in _lines
        assert 'Time: 600.0' in _lines
        assert 'Time: 900.0' in _lines
        assert 'Time: 1200.0' not in _lines


def test_py_hlvl_mtrx_1list_tsteps_cfg(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'timesteps', 3)
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_strata', True)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload'],
                                   [[0.2, 0.6]])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p)
    assert pp._has_matrix is True
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    pp.run_jobs()
    assert len(pp.file_list) == 2
    assert pp._is_completed is True
    f_bedload_list = [j.deltamodel.f_bedload for j in pp.job_list]
    assert sum([j == 0.2 for j in f_bedload_list]) == 1
    assert sum([j == 0.6 for j in f_bedload_list]) == 1
    end_time_000 = utilities.read_endtime_from_log(
        tmp_path / 'test' / 'job_000')
    end_time_001 = utilities.read_endtime_from_log(
        tmp_path / 'test' / 'job_001')
    assert end_time_000 == 900.0
    assert end_time_001 == 900.0
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc1 = os.path.join(
        tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc1)
    ds = netCDF4.Dataset(exp_path_nc0, "r", format="NETCDF4")
    assert ds.variables['strata_sand_frac'].shape[1:] == (10, 10)
    assert ds.variables['strata_sand_frac'].shape[1:] == (10, 10)
    assert ds.variables['strata_age'].shape == (4,)


def test_py_hlvl_mtrx_2list(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload', 'u0'],
                                   [[0.2, 0.5, 0.6], [1.0, 1.5, 2.0]])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    assert pp._has_matrix is True
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 9
    assert pp._is_completed is False
    pp.run_jobs()
    f_bedload_list = [j.deltamodel.f_bedload for j in pp.job_list]
    assert sum([j == 0.2 for j in f_bedload_list]) == 3
    assert sum([j == 0.5 for j in f_bedload_list]) == 3
    assert sum([j == 0.6 for j in f_bedload_list]) == 3
    comb_list = [(j.deltamodel.f_bedload, j.deltamodel.u0)
                 for j in pp.job_list]
    assert (0.2, 2.0) in comb_list
    assert (0.5, 1.0) in comb_list
    assert not (0.5, 0.2) in comb_list
    assert pp.job_list[0].deltamodel.dt == 300.0
    end_time_000 = utilities.read_endtime_from_log(
        tmp_path / 'test' / 'job_000')
    assert end_time_000 == 900.0
    assert len(pp.file_list) == 9
    assert pp._is_completed is True
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc5 = os.path.join(
        tmp_path / 'test', 'job_005', 'pyDeltaRCM_output.nc')
    exp_path_nc8 = os.path.join(
        tmp_path / 'test', 'job_008', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc5)
    assert os.path.isfile(exp_path_nc8)


def test_py_hlvl_mtrx_scientificnotation(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload', 'SLR'],
                                   [[0.2, 0.5, 0.6], [0.00004, 1e-6]])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3, dry_run=True)
    pp.run_jobs()
    SLR_list = [j.deltamodel.SLR for j in pp.job_list]
    assert sum([j == 4e-5 for j in SLR_list]) == 3
    assert sum([j == 0.000001 for j in SLR_list]) == 3


def test_py_hlvl_mtrx_one_list_time_config(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'time', 1000)
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload'],
                                   [[0.2, 0.6]])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p)
    assert pp._is_completed is False
    pp.run_jobs()
    assert pp._is_completed is True
    end_time_000 = utilities.read_endtime_from_log(
        tmp_path / 'test' / 'job_000')
    end_time_001 = utilities.read_endtime_from_log(
        tmp_path / 'test' / 'job_001')
    assert end_time_000 == 1200.0
    assert end_time_001 == 1200.0


def test_python_highlevelapi_matrix_needs_out_dir(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    # missing out_dir in the config will throw an error
    #   ==>  utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload'],
                                   [[0.2, 0.5, 0.6]])
    f.close()
    with pytest.raises(ValueError, match=r'You must specify "out_dir" in YAML .*'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_python_highlevelapi_matrix_outdir_exists_error(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_strata', 300)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload'],
                                   [[0.2, 0.5, 0.6]])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    with pytest.raises(FileExistsError, match=r'Job output directory .*'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_py_hlvl_mtrx_bad_type(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    # bad configuration will lead to error
    utilities.write_matrix_to_file(f,
                                   ['f_bedload', 'u0'],
                                   [[0.2, 0.5, 0.6], ['badstr1', 'badstr2']])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    with pytest.raises(TypeError, match='Input for "u0" not of the right type .*'):
        pp.run_jobs()


def test_py_hlvl_mtrx_bad_len1(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    # bad configuration is a list of length 1
    utilities.write_matrix_to_file(f,
                                   ['f_bedload', 'u0'],
                                   [[0.2], [0.5, 0.6, 1.25]])
    f.close()
    with pytest.raises(ValueError,  match=r'Length of matrix key "f_bedload" was 1,'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_py_hlvl_mtrx_bad_listinlist(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    # bad configuration will lead to error
    utilities.write_matrix_to_file(f,
                                   ['f_bedload', 'u0'],
                                   [[0.2, [0.5, 0.6]], [0.5, 1.25]])
    f.close()
    with pytest.raises(ValueError,  match=r'Depth of matrix expansion must not be > 1'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_py_hlvl_mtrx_bad_samekey(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.3)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    # bad configuration will lead to error
    utilities.write_matrix_to_file(f,
                                   ['f_bedload', 'u0'],
                                   [[0.2, 0.5, 0.6], [0.5, 1.25]])
    f.close()
    with pytest.raises(ValueError,  match=r'You cannot specify the same key in the matrix .*'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_py_hlvl_mtrx_bad_colon(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    # bad configuration will lead to error
    utilities.write_matrix_to_file(f,
                                   ['f_bedload', 'u0:'],
                                   [[0.2, 0.5], [0.5, 1.25]])
    f.close()
    with pytest.raises(ValueError,  match=r'Colon operator found in matrix expansion key.'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_py_hlvl_mtrx_no_out_dir_in_mtrx(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['out_dir', 'f_bedload'],
                                   [['dir1', 'dir2'], [0.2, 0.5, 0.6]])
    f.close()
    with pytest.raises(ValueError, match=r'You cannot specify "out_dir" as .*'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_python_highlevelapi_matrix_verbosity(tmp_path, capsys):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload', 'u0'],
                                   [[0.2, 0.5, 0.6], [1.5, 2.0]])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    captd = capsys.readouterr()
    assert 'Timestep: 0.0' not in captd.out
    assert 'Writing YAML file for job 0' in captd.out
    assert 'Writing YAML file for job 1' in captd.out
    assert 'Writing YAML file for job 2' in captd.out
    assert 'Writing YAML file for job 3' in captd.out
    assert 'Writing YAML file for job 4' in captd.out
    assert 'Writing YAML file for job 5' in captd.out
    assert 'Matrix expansion:' in captd.out
    assert '  dims 2' in captd.out
    assert '  jobs 6' in captd.out


def test_py_hlvl_ensemble(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    # assertions for job creation
    assert pp._has_matrix is True
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    pp.run_jobs()
    # assertions after running jobs
    assert len(pp.file_list) == 2
    assert pp._is_completed is True
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc1 = os.path.join(
        tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc1)


def test_py_hlvl_ensemble_with_matrix(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['f_bedload'],
                                   [[0.2, 0.5, 0.6]])
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    # assertions for job creation
    assert pp._has_matrix is True
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 6
    assert pp._is_completed is False
    pp.run_jobs()
    # assertions after running jobs
    assert len(pp.file_list) == 6
    assert pp._is_completed is True
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc5 = os.path.join(
        tmp_path / 'test', 'job_005', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc5)


def test_py_hlvl_parallel_boolean(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'parallel', True)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    # assertions for job creation
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    # assertions after running jobs
    if platform.system() == 'Linux':
        pp.run_jobs()
        assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
        assert pp._is_completed is True
        exp_path_nc0 = os.path.join(
            tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_nc1 = os.path.join(
            tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc0)
        assert os.path.isfile(exp_path_nc1)
    else:
        with pytest.raises(NotImplementedError,
                           match=r'Parallel simulations *.'):
            pp.run_jobs()


def test_py_hlvl_parallel_integer(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'parallel', 2)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    # assertions for job creation
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    # assertions after running jobs
    if platform.system() == 'Linux':
        pp.run_jobs()
        assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
        assert pp._is_completed is True
        exp_path_nc0 = os.path.join(
            tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_nc1 = os.path.join(
            tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc0)
        assert os.path.isfile(exp_path_nc1)
    else:
        with pytest.raises(NotImplementedError,
                           match=r'Parallel simulations *.'):
            pp.run_jobs()
    # NOTE: this does not actually test that
    #       *exactly* two jobs were run in parallel.


def test_py_hlvl_parallel_float(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'parallel', 2.3)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=3)
    # assertions for job creation
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    # assertions after running jobs
    if platform.system() == 'Linux':
        pp.run_jobs()
        assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
        assert pp._is_completed is True
        exp_path_nc0 = os.path.join(
            tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
        exp_path_nc1 = os.path.join(
            tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc0)
        assert os.path.isfile(exp_path_nc1)
    else:
        with pytest.raises(NotImplementedError,
                           match=r'Parallel simulations *.'):
            pp.run_jobs()
    # NOTE: this does not actually test that
    #       *exactly* two jobs were run in parallel.


@pytest.mark.skipif(platform.system() != 'Linux', reason='Parallel support \
                    only on Linux OS.')
def test_py_hlvl_parallel_checkpoint(tmp_path):
    """Test checkpointing in parallel."""
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'verbose', 2)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'parallel', 2)
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(f, 'save_eta_grids', True)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
    # assertions for job creation
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    # assertions after running jobs
    pp.run_jobs()
    assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
    assert pp._is_completed is True
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc1 = os.path.join(
        tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc1)
    # check that checkpoint files exist
    exp_path_ckpt0 = os.path.join(
        tmp_path / 'test', 'job_000', 'checkpoint.npz')
    exp_path_ckpt1 = os.path.join(
        tmp_path / 'test', 'job_001', 'checkpoint.npz')
    assert os.path.isfile(exp_path_ckpt0)
    assert os.path.isfile(exp_path_ckpt1)
    # load one output files and check values
    out_old = netCDF4.Dataset(exp_path_nc1)
    assert 'meta' in out_old.groups.keys()
    assert out_old['time'][0].tolist() == 0.0
    assert out_old['time'][-1].tolist() == 600.0  # 2 timesteps of 300.0s run

    # close netCDF file
    out_old.close()
    # try to resume jobs
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'verbose', 2)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'parallel', 2)
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'resume_checkpoint', True)
    utilities.write_parameter_to_file(f, 'save_eta_grids', True)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=1)
    # assertions for job creation
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    # assertions after running jobs
    pp.run_jobs()
    assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
    assert pp._is_completed is True
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc1 = os.path.join(
        tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc1)
    # check that checkpoint files still exist
    exp_path_ckpt0 = os.path.join(
        tmp_path / 'test', 'job_000', 'checkpoint.npz')
    exp_path_ckpt1 = os.path.join(
        tmp_path / 'test', 'job_001', 'checkpoint.npz')
    assert os.path.isfile(exp_path_ckpt0)
    assert os.path.isfile(exp_path_ckpt1)
    # load one output file to check it out
    out_fin = netCDF4.Dataset(exp_path_nc1)
    assert 'meta' in out_old.groups.keys()
    assert out_fin['time'][0].tolist() == 0
    assert out_fin['time'][-1].tolist() == 900.0  # +1 timestep of 300.0s
    # close netcdf file
    out_fin.close()


def test_py_hlvl_ensemble_badtype(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'ensemble', 2.0)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    with pytest.raises(TypeError, match=r'Invalid ensemble type, must be an integer.'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_py_hlvl_ensemble_double_seeds(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'seed', 1)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    with pytest.raises(ValueError,
                       match=r'You cannot specify the same key in the matrix '
                             'configuration and fixed configuration. Key "seed" '
                             'was specified in both.'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_py_hlvl_ensemble_matrix_seeds(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_matrix_to_file(f,
                                   ['seed'],
                                   [[1, 2]])
    f.close()
    with pytest.raises(ValueError,
                       match=r'Random seeds cannot be specified in the matrix, '
                             'if an "ensemble" number is specified as well.'):
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)


def test_Preprocessor_toplevelimport():
    import pyDeltaRCM

    assert 'Preprocessor' in dir(pyDeltaRCM)
    assert pyDeltaRCM.Preprocessor is pyDeltaRCM.preprocessor.Preprocessor


def test_subsidence_bounds(tmp_path):
    """Test subsidence bounds."""

    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 600.)
    utilities.write_parameter_to_file(f, 'Width', 600.)
    utilities.write_parameter_to_file(f, 'dx', 5)
    utilities.write_parameter_to_file(f, 'toggle_subsidence', True)
    utilities.write_parameter_to_file(f, 'theta1', -np.pi / 2)
    utilities.write_parameter_to_file(f, 'theta1', 0)
    f.close()
    with pytest.warns(UserWarning):
        delta = DeltaModel(input_file=p)
    # assert subsidence mask is binary
    assert np.all(delta.subsidence_mask == delta.subsidence_mask.astype(bool))
    # check specific regions
    assert np.all(delta.subsidence_mask[75:, 60:] == 1)
    assert np.all(delta.subsidence_mask[:, :55] == 0)


# test private variable setters for valid values

def test_negative_length(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Length', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_width(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Width', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_dx(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'dx', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_bigger_than_Width_dx(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Width', 10)
    utilities.write_parameter_to_file(f, 'dx', 100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_bigger_than_Length_dx(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Length', 10)
    utilities.write_parameter_to_file(f, 'dx', 100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_L0_meters(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'L0_meters', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_itermax(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'itermax', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_Np_water(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Np_water', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_N0_meters(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'N0_meters', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_Np_sed(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Np_sed', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_f_bedload(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'f_bedload', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_big_f_bedload(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'f_bedload', 2)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_C0_percent(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'C0_percent', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


def test_negative_Csmooth(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'Csmooth', -100)
    f.close()
    with pytest.raises(ValueError):
        delta = DeltaModel(input_file=p)


class TestScaleRelativeSeaLeveLRiseRate():

    def test_scale_If_1(self):
        scaled = preprocessor.scale_relative_sea_level_rise_rate(5, If=1)
        assert scaled == 5 / 1000 / 365.25 / 86400

    def test_scale_If_0p1(self):
        scaled = preprocessor.scale_relative_sea_level_rise_rate(5, If=0.1)
        assert scaled == 5 / 1000 / 365.25 / 86400 * 10


def test_load_nocheckpoint(tmp_path):
    """Try loading a checkpoint file when one doesn't exist."""
    # define a yaml
    file_name = 'trial_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
    base_f.close()

    # try loading the model yaml despite no checkpoint existing
    with pytest.raises(FileNotFoundError):
        _ = DeltaModel(input_file=base_p)
