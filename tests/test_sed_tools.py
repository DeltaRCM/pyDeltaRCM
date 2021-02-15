# unit tests for sed_tools.py

import pytest

import sys
import os
import numpy as np

from .utilities import test_DeltaModel


def test_sed_route_padding(test_DeltaModel):
    """
    test the function sed_tools.sed_route
    """
    test_DeltaModel.pad_cell_type = np.pad(
        test_DeltaModel.cell_type, 1, 'edge')
    test_DeltaModel.pad_stage = np.pad(test_DeltaModel.stage, 1, 'edge')
    test_DeltaModel.pad_depth = np.pad(test_DeltaModel.depth, 1, 'edge')

    test_DeltaModel.pad_cell_type = np.pad(
        test_DeltaModel.cell_type, 1, 'edge')
    test_DeltaModel.sed_route()

    assert np.all(test_DeltaModel.pad_depth[
                  1:-1, 1:-1] == test_DeltaModel.depth)
    assert np.shape(test_DeltaModel.pad_depth) == (12, 12)


def test_sand_route_updates(test_DeltaModel):
    # operations at top of sed_route()
    test_DeltaModel.pad_depth = np.pad(test_DeltaModel.depth, 1, 'edge')
    test_DeltaModel.qs[:] = 0
    test_DeltaModel.Vp_dep_sand[:] = 0
    test_DeltaModel.Vp_dep_mud[:] = 0

    test_DeltaModel.pad_cell_type = np.pad(
        test_DeltaModel.cell_type, 1, 'edge')
    test_DeltaModel.pad_stage = np.pad(test_DeltaModel.stage, 1, 'edge')
    test_DeltaModel.pad_depth = np.pad(test_DeltaModel.depth, 1, 'edge')

    # operations at the
    test_DeltaModel.pad_cell_type = np.pad(
        test_DeltaModel.cell_type, 1, 'edge')

    test_DeltaModel.route_all_sand_parcels()

    # simply check that the sediment transport field is updated
    assert np.any(test_DeltaModel.qs != 0)
