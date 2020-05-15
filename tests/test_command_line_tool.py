
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
    printed = subprocess.run(['run_pyDeltaRCM', '--version'], capture_output=True, text=True)
    assert printed.stdout == 'pyDeltaRCM ' + deltaModule.__version__ + '\n'


# def test_run_model():
#     """
#     test calling the command line feature with a config file.
#     """
#     command_line.run_model(use_test_yaml=True)
#     assert os.path.isfile(os.path.join(os.getcwd(), 'test', 'eta_0.0.png'))
#     shutil.rmtree(os.path.join(os.getcwd(), 'test'))
