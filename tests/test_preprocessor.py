
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
    utilities.write_parameter_to_file(f, 'S0', 0.005)
    utilities.write_parameter_to_file(f, 'timesteps', 1)
    utilities.write_parameter_to_file(f, 'out_dir', 'test')
    utilities.write_parameter_to_file(f, 'save_dt', 1)
    f.close()
    subprocess.check_output(['pyDeltaRCM',
                             '--config', p,
                             '--dryrun'])
    exp_path = os.path.join(os.getcwd(), 'test', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path)


def test_entry_point_python_main_call(tmp_path):
    """
    test calling the python hook command line feature with a config file.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'S0', 0.005)
    utilities.write_parameter_to_file(f, 'timesteps', 1)
    utilities.write_parameter_to_file(f, 'out_dir', 'test')
    f.close()
    subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                             '--config', p,
                             '--dryrun'])
    exp_path = os.path.join(os.getcwd(), 'test', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path)


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
