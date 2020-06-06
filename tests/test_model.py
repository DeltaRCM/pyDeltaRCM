# unit tests for deltaRCM_driver.py

import pytest

import sys
import os
import numpy as np

from utilities import test_DeltaModel

# need to create a simple case of pydeltarcm object to test these functions


def test_init(test_DeltaModel):
    """
    test the deltaRCM_driver init (happened when delta.initialize was run)
    """
    assert test_DeltaModel._time == 0.
    assert test_DeltaModel._is_finalized == False


def test_update(test_DeltaModel):
    test_DeltaModel.update()
    assert test_DeltaModel._time == 1.0
    assert test_DeltaModel._is_finalized == False


def test_finalize(test_DeltaModel):
    test_DeltaModel.update()
    test_DeltaModel.finalize()
    assert test_DeltaModel._is_finalized == True


def test_multifinalization_error(test_DeltaModel):
    err_delta = test_DeltaModel
    err_delta.update()
    # test will fail if any assertion is wrong
    assert err_delta._time == 1.0 
    assert err_delta._is_finalized == False
    err_delta.finalize()
    assert err_delta._is_finalized == True
    # next line should throw RuntimeError
    with pytest.raises(RuntimeError):
        err_delta.update()


def test_initial_values(test_DeltaModel):

    # delta = DeltaModel(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))
    assert np.all(test_DeltaModel.sea_surface_elevation == 0)
    assert test_DeltaModel.water_depth[0, 2] == 0
    assert test_DeltaModel.water_depth[0, 3] == 1
    assert test_DeltaModel.water_depth[4, 4] == 1
    assert test_DeltaModel.bed_elevation[0, 2] == 0
    assert test_DeltaModel.bed_elevation[0, 3] == -1
    assert test_DeltaModel.bed_elevation[4, 4] == -1


def test_setting_getting_sea_surface_mean_elevation(test_DeltaModel):
    assert test_DeltaModel.sea_surface_mean_elevation == 0
    test_DeltaModel.sea_surface_mean_elevation = 0.5
    assert test_DeltaModel.sea_surface_mean_elevation == 0.5


def test_setting_getting_sea_surface_elevation_change(test_DeltaModel):
    assert test_DeltaModel.sea_surface_elevation_change == 0.001
    test_DeltaModel.sea_surface_elevation_change = 0.002
    assert test_DeltaModel.sea_surface_elevation_change == 0.002


def test_setting_getting_bedload_fraction(test_DeltaModel):
    assert test_DeltaModel.bedload_fraction == 0.5
    test_DeltaModel.bedload_fraction = 0.25
    assert test_DeltaModel.bedload_fraction == 0.25


def test_setting_getting_channel_flow_velocity(test_DeltaModel):
    assert test_DeltaModel.channel_flow_velocity == 1
    test_DeltaModel.channel_flow_velocity = 3
    assert test_DeltaModel.channel_flow_velocity == 3
    assert test_DeltaModel.u_max == 6


def test_setting_getting_channel_width(test_DeltaModel):
    assert test_DeltaModel.channel_width == 2
    test_DeltaModel.channel_width = 10
    assert test_DeltaModel.channel_width == 10
    assert test_DeltaModel.N0 == 10


def test_setting_getting_channel_width(test_DeltaModel):
    assert test_DeltaModel.channel_flow_depth == 1
    test_DeltaModel.channel_flow_depth = 2
    assert test_DeltaModel.channel_flow_depth == 2

    
def test_setting_getting_channel_width(test_DeltaModel):
    assert test_DeltaModel.influx_sediment_concentration == 0.1
    test_DeltaModel.influx_sediment_concentration = 2
    assert test_DeltaModel.C0 == 0.02
