
import pytest

import os
import shutil
import locale
import numpy as np
import subprocess

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM import shared_tools

import utilities
from utilities import test_DeltaModel


# test yaml parsing

def test_override_from_testfile(test_DeltaModel):
    out_path = test_DeltaModel.out_dir.split('/')
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


def test_random_seed_settings_noaction_default(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'S0', 0.005)
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.seed is None
    assert np.random.seed is not 0


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
    utilities.write_parameter_to_file(f, 'timesteps', 2)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    subprocess.check_output(['pyDeltaRCM',
                             '--config', p])
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
    utilities.write_parameter_to_file(f, 'timesteps', 1)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                             '--config', p])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png)
    assert not os.path.isfile(exp_path_png1)


def test_entry_point_python_main_call_dryrun(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'timesteps', 1)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                             '--config', p,
                             '--dryrun'])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
    assert os.path.isfile(exp_path_nc)
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
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                             '--config', p,
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
                                 '--config', p])


import pyDeltaRCM as _pyimportedalias


def test_version_call():
    """
    test calling the command line feature to query the version.
    """
    encoding = locale.getpreferredencoding()
    printed1 = subprocess.run(['pyDeltaRCM', '--version'],
                              stdout=subprocess.PIPE, encoding=encoding)
    assert printed1.stdout == 'pyDeltaRCM ' + _pyimportedalias.__version__ + '\n'
    printed2 = subprocess.run(
        ['python', '-m', 'pyDeltaRCM', '--version'], stdout=subprocess.PIPE, encoding=encoding)
    assert printed2.stdout == 'pyDeltaRCM ' + _pyimportedalias.__version__ + '\n'


# test high level python api
from pyDeltaRCM import preprocessor


def test_python_highlevelapi_call_without_args():
    with pytest.raises(ValueError):
        pp = preprocessor.Preprocessor()


def test_python_highlevelapi_call_without_timesteps(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    with pytest.raises(ValueError):
        pp = preprocessor.Preprocessor(p)


def test_python_highlevelapi_call_with_timesteps_yaml_init_types(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'timesteps', 2)
    f.close()
    pp = preprocessor.Preprocessor(p)
    assert type(pp.job_list) is list
    assert len(pp.job_list) == 1
    assert type(pp.job_list[0]) is preprocessor.Preprocessor._Job
    assert type(pp.job_list[0].deltamodel) is DeltaModel
    assert pp.job_list[0]._is_completed == False


def test_python_highlevelapi_call_with_timesteps_yaml_runjobs(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'timesteps', 2)
    f.close()
    pp = preprocessor.Preprocessor(p)
    assert len(pp.job_list) == 1
    assert pp.job_list[0]._is_completed == False
    pp.run_jobs()
    assert pp.job_list[0]._is_completed == True


def test_python_highlevelapi_call_with_args(tmp_path):
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
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    utilities.write_parameter_to_file(f, 'save_eta_figs', True)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
    assert type(pp.job_list) is list
    assert len(pp.job_list) == 1
    assert type(pp.job_list[0]) is preprocessor.Preprocessor._Job
    assert type(pp.job_list[0].deltamodel) is DeltaModel
    assert pp.job_list[0].deltamodel.Length == 10.0
    assert pp.job_list[0].deltamodel.Width == 10.0
    assert pp.job_list[0].deltamodel.dx == 1.0
    assert pp.job_list[0].deltamodel.seed == 0
    assert pp.job_list[0]._is_completed == False
    pp.run_jobs()
    assert len(pp.job_list) == 1
    assert pp.job_list[0]._is_completed == True
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
    exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
    exp_path_png3 = os.path.join(tmp_path / 'test', 'eta_00002.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png)
    assert os.path.isfile(exp_path_png1)
    assert not os.path.isfile(exp_path_png3)


def test_Preprocessor_toplevelimport():
    import pyDeltaRCM

    assert 'Preprocessor' in dir(pyDeltaRCM)
    assert pyDeltaRCM.Preprocessor is pyDeltaRCM.preprocessor.Preprocessor
