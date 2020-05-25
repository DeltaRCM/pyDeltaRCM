# unit tests for consistent model outputs

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM

# need to create a simple case of pydeltarcm object to test these functions


def test_bed_after_one_update():
    """
    """
    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))
    delta.update()
    # slice is: delta.eta[:5, 4]
    # print(delta.eta[:5, 4])

    _exp = np.array([-1., -0.840265, -0.9976036, -1., -1.])
    assert np.all(delta.eta[:5, 4] == pytest.approx(_exp))
