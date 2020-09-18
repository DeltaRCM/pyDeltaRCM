
import pytest
import utilities
import os


def test_cli_notimestep(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    result = os.system("pyDeltaRCM --config " + str(p))
    # if the result is 256 this is an error code
    assert result == 256


def test_cli_noconfig():
    result = os.system("pyDeltaRCM --timesteps 1")
    assert result == 0


def test_cli_noargs():
    result = os.system("pyDeltaRCM")
    # returns an error code
    assert result == 256
