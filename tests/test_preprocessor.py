
import pytest

import os
import shutil
import locale
import numpy as np
import subprocess

import pyDeltaRCM as _pyimportedalias
from pyDeltaRCM import preprocessor


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
    encoding = locale.getpreferredencoding()
    printed1 = subprocess.run(['run_pyDeltaRCM', '--version'], stdout=subprocess.PIPE, encoding=encoding)
    assert printed1.stdout == _pyimportedalias.__version__ + '\n'
    printed2 = subprocess.run(['python', '-m', 'pyDeltaRCM', '--version'], stdout=subprocess.PIPE, encoding=encoding)
    assert printed2.stdout == _pyimportedalias.__version__ + '\n'


def test_python_call():
    """
    test calling the python hook command line feature with a config file.
    """
    subprocess.run(['python', '-m', 'pyDeltaRCM', '--config', os.path.join(os.getcwd(), 'tests', 'test_output.yaml')])
    assert os.path.isfile(os.path.join(os.getcwd(), 'test', 'eta_0.0.png'))
    shutil.rmtree(os.path.join(os.getcwd(), 'test'))
