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
    print(delta.eta[:5, 62])

    _exp1 = np.array([-4.9709163, -4.972, -3.722989, -7.786886, -3.7249935])
    assert np.all(delta.eta[:5, 62] == pytest.approx(_exp1))

    for _ in range(0, 10):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 4]
    print(delta.eta[:5, 62])

    _exp2 = np.array([-4.9709163, -1.5536911, -3.268889, -3.2696986, -2.0843806])
    assert np.all(delta.eta[:5, 62] == pytest.approx(_exp2))


def test_limit_inds_error_inlet_size_fixed_bug_example_1(tmp_path):
    """IndexError due to inlet size being too large by default.

    If the domain was made small (30x60), but the `N0_meters` and `L0_meters`
    parameters were not adjusted, the model domain was filled with landscape
    above sea level and water routing failed due to trying to access a cell
    outside the domain. This produced an IndexError.

    We now limit the size of the inlet to 1/4 of the domain edge length in
    both directions (length and width). This change fixed this case example,
    and it is now here as a consistency check, and to make sure the bug is not
    recreated by mistake.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 42)
    utilities.write_parameter_to_file(f, 'Length', 30.)
    utilities.write_parameter_to_file(f, 'Width', 60.)
    utilities.write_parameter_to_file(f, 'dx', 1)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.65)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 2):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 30]
    print(delta.eta[:5, 30])

    _exp = np.array([-4.9988008, -4.7794013, -4.5300136, -4.4977293, -4.56228])
    assert np.all(delta.eta[:5, 30] == pytest.approx(_exp))


def test_limit_inds_error_inlet_size_fixed_bug_example_2(tmp_path):
    """IndexError due to inlet size being too large by default.

    If the domain was made small (30x60), but the `N0_meters` and `L0_meters`
    parameters were not adjusted, the model domain was filled with landscape
    above sea level and water routing failed due to trying to access a cell
    outside the domain. This produced an IndexError.

    We now limit the size of the inlet to 1/4 of the domain edge length in
    both directions (length and width). This change fixed this case example,
    and it is now here as a consistency check, and to make sure the bug is not
    recreated by mistake.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 43)
    utilities.write_parameter_to_file(f, 'Length', 30.)
    utilities.write_parameter_to_file(f, 'Width', 60.)
    utilities.write_parameter_to_file(f, 'dx', 1)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.15)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 7):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 30]
    print(delta.eta[:5, 30])

    _exp = np.array([-4.9975486, -4.9140935, -5.15276, -5.3690896, -5.1903167])
    assert np.all(delta.eta[:5, 30] == pytest.approx(_exp))


def test_limit_inds_error_fixed_bug_example_3(tmp_path):
    """IndexError due to inlet width resolving to an edge cell.

    If the domain was made small and long (20x10), then the configuration that
    determined the center cell to hinge the inlet location on, would resolve
    to place the inlet at an edge cell. This led to some index error at the
    end of the water iteration and an IndexError.

    We now recalcualte the value of the self.CTR parameter if the cell is
    chosen as ind = 0 or 1. This is really only relevant to the testing suite,
    where domains are very small. This change fixed this case example, and it
    is now here as a consistency check, and to make sure the bug is not
    recreated by mistake.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 42)
    utilities.write_parameter_to_file(f, 'Length', 20.)
    utilities.write_parameter_to_file(f, 'Width', 10.)
    utilities.write_parameter_to_file(f, 'dx', 2)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.65)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 2):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 2]
    print(delta.eta[:5, 2])

    _exp = np.array([-4.99961, -4.605685, -3.8314152, -4.9007816, -5.])
    assert np.all(delta.eta[:5, 2] == pytest.approx(_exp))
