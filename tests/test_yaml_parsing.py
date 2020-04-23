import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM


# utilities for file writing
def create_temporary_file(tmp_path, file_name):
    d = tmp_path / 'configs'
    d.mkdir()
    p = d / file_name
    f = open(p, "a")
    return p, f


def write_parameter_to_file(f, varname, varvalue):
    f.write(varname + ': ' + str(varvalue) + '\n')


# tests
def test_with_no_argument():
    delta = pyDeltaRCM()
    assert delta.out_dir == 'deltaRCM_Output'
    assert delta.Length == 1000


def test_override_from_testfile():
    delta = pyDeltaRCM(input_file=os.path.join(
        os.getcwd(), 'tests', 'test.yaml'))
    assert delta.out_dir == 'test'
    assert delta.Length == 10


@pytest.mark.xfail(raises=FileNotFoundError)
def test_error_if_no_file_found():
    delta = pyDeltaRCM(input_file='./nonexisting_file.yaml')


def test_override_single_default(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = create_temporary_file(tmp_path, file_name)
    write_parameter_to_file(f, 'S0', 0.005)
    f.close()
    delta = pyDeltaRCM(input_file=p)
    assert delta.S0 == 0.005


def test_override_two_defaults(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = create_temporary_file(tmp_path, file_name)
    write_parameter_to_file(f, 'S0', 0.005)
    write_parameter_to_file(f, 'Np_sed', 2)
    f.close()
    delta = pyDeltaRCM(input_file=p)
    assert delta.S0 == 0.005
    assert delta.Np_sed == 2


@pytest.mark.xfail(raises=TypeError)
def test_override_bad_type_float_string(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = create_temporary_file(tmp_path, file_name)
    write_parameter_to_file(f, 'S0', 'a string?!')
    f.close()
    delta = pyDeltaRCM(input_file=p)


@pytest.mark.xfail(raises=TypeError)
def test_override_bad_type_float_int(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = create_temporary_file(tmp_path, file_name)
    write_parameter_to_file(f, 'S0', 2)
    f.close()
    delta = pyDeltaRCM(input_file=p)


@pytest.mark.xfail(raises=TypeError)
def test_override_bad_type_int_float(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = create_temporary_file(tmp_path, file_name)
    write_parameter_to_file(f, 'beta', 24.4234)
    f.close()
    delta = pyDeltaRCM(input_file=p)


def test_not_creating_illegal_attributes(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = create_temporary_file(tmp_path, file_name)
    write_parameter_to_file(f, 'illegal_attribute', True)
    f.close()
    delta = pyDeltaRCM(input_file=p)
    assert delta.S0 == 0.0002  # from default.yaml
    assert not hasattr(delta, 'illegal_attribute')


def test_not_overwriting_existing_attributes(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = create_temporary_file(tmp_path, file_name)
    write_parameter_to_file(f, 'input_file', '/fake/path.yaml')
    f.close()
    delta = pyDeltaRCM(input_file=p)
    assert delta.S0 == 0.0002  # from default.yaml
    assert hasattr(delta, 'input_file')
    assert delta.input_file == p
