# unit tests for deltaRCM_tools.py

import pytest

import sys
import os
import numpy as np

from utilities import test_DeltaModel


def test_run_one_timestep(test_DeltaModel):
    test_DeltaModel.run_one_timestep()
    # basically assume sediment has been added at inlet
    assert test_DeltaModel.qs[0, 4] != 0.


def test_finalize_timestep(test_DeltaModel):
    test_DeltaModel.finalize_timestep()
    # check that sea level rose as expected
    assert test_DeltaModel.H_SL == 0.3
