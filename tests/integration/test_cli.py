import pytest
import os
import platform
import numpy as np

from ..utilities import test_DeltaModel
from .. import utilities
from pyDeltaRCM import DeltaModel


class TestCommandLineInterfaceDirectly:

    def test_cli_notimestep(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        result = os.system("pyDeltaRCM --config " + str(p))
        # if the result is 256 or 1 this is an error code
        if platform.system() == 'Windows':
            assert result == 1
        else:
            assert result == 256

    def test_cli_noconfig(self):
        result = os.system("pyDeltaRCM --timesteps 1")
        assert result == 0

    def test_cli_noargs(self):
        result = os.system("pyDeltaRCM")
        # returns an error code
        if platform.system() == 'Windows':
            assert result == 1
        else:
            assert result == 256


class TestConsistentOutputsBetweenMerges:

    def test_bed_after_one_update(self, test_DeltaModel):

        test_DeltaModel.update()

        # slice is: test_DeltaModel.eta[:5, 4]
        _exp = np.array([-1., -0.9152762, -1.0004134, -1., -1.])
        assert np.all(test_DeltaModel.eta[:5, 4] == pytest.approx(_exp))

    def test_bed_after_ten_updates(self, test_DeltaModel):

        for _ in range(0, 10):
            test_DeltaModel.update()

        # slice is: test_DeltaModel.eta[:5, 4]
        _exp = np.array([1.7, 0.394864, -0.95006764,  -1., -1.])
        assert np.all(test_DeltaModel.eta[:5, 4] == pytest.approx(_exp))

    def test_long_multi_validation(self, tmp_path):
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
        _exp1 = np.array([-4.976912, -4.979, -7.7932253, -4.9805, -2.7937498])
        assert np.all(delta.eta[:5, 62] == pytest.approx(_exp1))

        for _ in range(0, 10):
            delta.update()

        _exp2 = np.array([-4.9614887, -3.4891236, -12.195051,  -4.6706524, -2.7937498])
        assert np.all(delta.eta[:5, 62] == pytest.approx(_exp2))
        delta.finalize()
