
import pytest

import os
import shutil
import locale
import numpy as np
import subprocess

import pyDeltaRCM as _pyimportedalias
from pyDeltaRCM import preprocessor

import utilities


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
    exp_path_png0 = os.path.join(tmp_path / 'test', 'eta_0.0.png')
    exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_1.0.png')
    assert os.path.isfile(exp_path_nc)
    assert os.path.isfile(exp_path_png0)
    assert os.path.isfile(exp_path_png1)


def test_entry_point_python_main_call(tmp_path):
    """
    test calling the python hook command line feature with a config file.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'timesteps', 1)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    f.close()
    subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                             '--config', p,
                             '--dryrun'])
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    exp_path_png = os.path.join(tmp_path / 'test', 'eta_0.0.png')
    assert os.path.isfile(exp_path_nc)
    assert not os.path.isfile(exp_path_png) # does not exist because --dryrun


# subprocess.CalledProcessError

def test_version_call():
    """
    test calling the command line feature to query the version.
    """
    encoding = locale.getpreferredencoding()
    printed1 = subprocess.run(
        ['pyDeltaRCM', '--version'], stdout=subprocess.PIPE, encoding=encoding)
    assert printed1.stdout == _pyimportedalias.__version__ + '\n'
    printed2 = subprocess.run(
        ['python', '-m', 'pyDeltaRCM', '--version'], stdout=subprocess.PIPE, encoding=encoding)
    assert printed2.stdout == _pyimportedalias.__version__ + '\n'
