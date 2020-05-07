# unit tests for consistent model outputs

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM

# need to create a simple case of pydeltarcm object to test these functions
delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))


def test_bed_after_one_update():
    """
    """
    delta.update()
    # slice is: delta.eta[:4, 2:7]
    _exp = np.array([[0.,         -1.,         -1.,         -1.,         0.],
                     [-0.8469125, -0.82371545, -0.82951754, -0.909775,  -0.9997975],
                     [-0.99919,   -0.99838,    -0.9981775,  -0.9989875, -0.9997975],
                     [-1.,        -1.,         -1.,         -1.,        -1.]])
    assert np.all(delta.eta[:4, 2:7] == pytest.approx(_exp))
