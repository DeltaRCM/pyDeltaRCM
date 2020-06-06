# unit tests for consistent model outputs

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM import DeltaModel

from utilities import test_DeltaModel
import utilities

# need to create a simple case of pydeltarcm object to test these functions


def test_bed_after_one_update(test_DeltaModel):
    test_DeltaModel.update()
    # slice is: test_DeltaModel.eta[:5, 4]
    # print(test_DeltaModel.eta[:5, 4])

    _exp = np.array([-1., -0.840265, -0.9976036, -1., -1.])
    assert np.all(test_DeltaModel.eta[:5, 4] == pytest.approx(_exp))


def test_bed_after_ten_updates(test_DeltaModel):

    for _ in range(0, 10):
        test_DeltaModel.update()

    # slice is: test_DeltaModel.eta[:5, 4]
    # print(test_DeltaModel.eta[:5, 4])

    _exp = np.array([1.7, 0.83358884, -0.9256229,  -1., -1.])
    assert np.all(test_DeltaModel.eta[:5, 4] == pytest.approx(_exp))


def test_long_multi_validation(tmp_path):
    # IndexError on corner.

    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 42)
    utilities.write_parameter_to_file(f, 'Length', 600.)
    utilities.write_parameter_to_file(f, 'Width', 600.)
    utilities.write_parameter_to_file(f, 'dx', 5)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.05)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 3):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 62]
    # print(delta.eta[:5, 62])

    _exp1 = np.array([-4.971009,  -3.722004,  -4.973,     -3.7240038, -3.7250037])
    assert np.all(delta.eta[:5, 62] == pytest.approx(_exp1))

    for _ in range(0, 30):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 4]
    print(delta.eta[:5, 62])

    _exp2 = np.array([-4.962428,  -1.3612521, -2.2904062, -1.4572337, -0.864957])
    assert np.all(delta.eta[:5, 62] == pytest.approx(_exp2))
