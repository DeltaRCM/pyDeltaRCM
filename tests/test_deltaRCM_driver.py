# unit tests for deltaRCM_driver.py

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM

# need to create a simple case of pydeltarcm object to test these functions
delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))


def test_init():
    """
    test the deltaRCM_driver init (happened when delta.initialize was run)
    """
    assert delta._time == 0.
    assert delta._is_finalized == False


def test_update():
    delta.update()
    assert delta._time == 1.0
    assert delta._is_finalized == False


def test_finalize():
    delta.finalize()
    assert delta._is_finalized == True


@pytest.mark.xfail(raises=RuntimeError, strict=True)
def test_multifinalization_error():
    err_delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))
    err_delta.update()
    # test will Fail if any assertion is wrong
    assert err_delta._time == 1.0 
    assert err_delta._is_finalized == False
    err_delta.finalize()
    assert err_delta._is_finalized == True
    # next line should throw RuntimeError and test will xFail
    err_delta.update()
