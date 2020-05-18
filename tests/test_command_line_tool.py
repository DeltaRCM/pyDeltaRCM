
import pytest

import os
import shutil
import numpy as np
import subprocess

import pyDeltaRCM as deltaModule
from pyDeltaRCM import command_line


def test_call():
    """
    test calling the command line feature with a config file.
    """
    subprocess.run(['run_pyDeltaRCM', '--config', os.path.join(os.getcwd(), 'tests', 'test_output.yaml')])
    assert os.path.isfile(os.path.join(os.getcwd(), 'test', 'eta_0.0.png'))
    shutil.rmtree(os.path.join(os.getcwd(), 'test'))


def test_version_call():
    """
    test calling the command line feature to query the version.
    """
    printed1 = subprocess.run(['run_pyDeltaRCM', '--version'], capture_output=True, text=True)
    assert printed1.stdout == 'pyDeltaRCM ' + deltaModule.__version__ + '\n'
    printed2 = subprocess.run(['python', '-m', 'pyDeltaRCM', '--version'], capture_output=True, text=True)
    assert printed2.stdout == 'pyDeltaRCM ' + deltaModule.__version__ + '\n'


def test_python_call():
    """
    test calling the python hook command line feature with a config file.
    """
    subprocess.run(['python', '-m', 'pyDeltaRCM', '--config', os.path.join(os.getcwd(), 'tests', 'test_output.yaml')])
    assert os.path.isfile(os.path.join(os.getcwd(), 'test', 'eta_0.0.png'))
    shutil.rmtree(os.path.join(os.getcwd(), 'test'))
