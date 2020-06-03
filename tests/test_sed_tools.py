# unit tests for sed_tools.py

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM
from pyDeltaRCM import sed_tools

# need to create a simple case of pydeltarcm object to test these functions
delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

# now that it is initiated can access the shared_tools via the inherited object
# delta._delta.**sed_tools_function**


def test_sed_route():
    """
    test the function sed_tools.sed_route
    """
    delta.pad_cell_type = np.pad(delta.cell_type, 1, 'edge')
    delta.pad_stage = np.pad(delta.stage, 1, 'edge')
    delta.pad_depth = np.pad(delta.depth, 1, 'edge')
    delta.sed_route()
    [a, b] = np.shape(delta.pad_depth)

    assert a == 12
