# unit tests for deltaRCM_driver.py

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM

# need to create a simple case of pydeltarcm object to test these functions


def test_init():
    """
    test the deltaRCM_driver init (happened when delta.initialize was run)
    """
    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))
    assert delta._time == 0.
    assert delta._is_finalized == False


def test_update():
    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

    delta.update()
    assert delta._time == 1.0
    assert delta._is_finalized == False


def test_finalize():
    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))
    delta.finalize()
    assert delta._is_finalized == True


def test_multifinalization_error():
    err_delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))
    err_delta.update()
    # test will Fail if any assertion is wrong
    assert err_delta._time == 1.0
    assert err_delta._is_finalized == False
    err_delta.finalize()
    assert err_delta._is_finalized == True
    # next line should throw RuntimeError
    with pytest.raises(RuntimeError):
        err_delta.update()


def test_getters_setters():

    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

    assert np.all(delta.sea_surface_elevation == 0)
    assert delta.water_depth[0, 2] == 0
    assert delta.water_depth[0, 3] == 1
    assert delta.water_depth[4, 4] == 1
    assert delta.bed_elevation[0, 2] == 0
    assert delta.bed_elevation[0, 3] == -1
    assert delta.bed_elevation[4, 4] == -1

    assert delta.sea_surface_mean_elevation == 0
    delta.sea_surface_mean_elevation = 0.5
    assert delta.sea_surface_mean_elevation == 0.5

    assert delta.sea_surface_elevation_change == 0.001
    delta.sea_surface_elevation_change = 0.002
    assert delta.sea_surface_elevation_change == 0.002

    assert delta.bedload_fraction == 0.5
    delta.bedload_fraction = 0.25
    assert delta.bedload_fraction == 0.25

    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

    assert delta.channel_flow_velocity == 1
    delta.channel_flow_velocity = 3
    assert delta.channel_flow_velocity == 3
    assert delta.u_max == 6

    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

    assert delta.channel_width == 2
    delta.channel_width = 10
    assert delta.channel_width == 10
    assert delta.N0 == 10

    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

    assert delta.channel_flow_depth == 1
    delta.channel_flow_depth = 2
    assert delta.channel_flow_depth == 2

    delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

    assert delta.influx_sediment_concentration == 0.1
    delta.influx_sediment_concentration = 2
    assert delta.C0 == 0.02
