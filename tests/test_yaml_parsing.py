import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.model import DeltaModel

import utilities


def test_with_no_argument():
    delta = DeltaModel()
    assert delta.out_dir == 'deltaRCM_Output'
    assert delta.Length == 1000


def test_override_from_testfile():
    delta = DeltaModel(input_file=os.path.join(
        os.getcwd(), 'tests', 'test.yaml'))
    assert delta.out_dir == 'test'
    assert delta.Length == 10


@pytest.mark.xfail(raises=FileNotFoundError, strict=True)
def test_error_if_no_file_found():
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


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_override_bad_type_float_string(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'S0', 'a string?!')
    f.close()
    delta = DeltaModel(input_file=p)


@pytest.mark.xfail(raises=TypeError, strict=True)
def test_override_bad_type_int_float(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'beta', 24.4234)
    f.close()
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


# paramter specific testing

def test_random_seed_settings_value(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    utilities.write_parameter_to_file(f, 'seed', 9999)
    f.close()
    np.random.seed(9999)
    _preval_same = np.random.uniform()
    np.random.seed(5)
    _preval_diff = np.random.uniform(1000)
    delta = DeltaModel(input_file=p)
    assert delta.seed == 9999
    _postval_same = np.random.uniform()
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
