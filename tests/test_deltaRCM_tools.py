# unit tests for deltaRCM_tools.py

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.model import DeltaModel

# need to create a simple case of pydeltarcm object to test these functions
delta = DeltaModel(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

# now that it is initiated can access the shared_tools via the inherited object
# delta._delta.**Tools_function**


def test_run_one_timestep():
    delta.run_one_timestep()
    # basically assume sediment has been added at inlet
    assert delta.qs[0, 4] != 0.


def test_finalize_timestep():
    delta.finalize_timestep()
    # check that sea level rose as expected
    assert delta.H_SL == 0.3
