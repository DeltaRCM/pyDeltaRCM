# unit tests for sed_tools.py

import numpy as np

import pytest
import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from . import utilities


class TestSedimentRoute:

    def test_route_sediment(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock top-level methods
        _delta.log_info = mock.MagicMock()
        _delta.init_sediment_iteration = mock.MagicMock()
        _delta.route_all_sand_parcels = mock.MagicMock()
        _delta.topo_diffusion = mock.MagicMock()
        _delta.route_all_mud_parcels = mock.MagicMock()

        # run the method
        _delta.route_sediment()

        # methods called
        assert (_delta.log_info.call_count == 4)
        assert (_delta.init_sediment_iteration.called is True)
        assert (_delta.route_all_sand_parcels.called is True)
        assert (_delta.topo_diffusion.called is True)
        assert (_delta.route_all_mud_parcels.called is True)

    def test_sed_route_deprecated(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # mock top-level methods
        _delta.logger = mock.MagicMock()
        _delta.route_sediment = mock.MagicMock()

        # check warning raised
        with pytest.warns(UserWarning):
            _delta.sed_route()

        # and logged
        assert (_delta.logger.warning.called is True)


class TestInitSedimentIteration:

    def test_fields_cleared(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # alter field for initial values going into function
        _delta.qs += np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.Vp_dep_sand += np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.Vp_dep_mud += np.random.uniform(0, 1, size=_delta.eta.shape)

        assert not np.all(_delta.qs == 0)  # field has info

        # call the method
        _delta.init_sediment_iteration()

        # assertions
        assert np.all(_delta.pad_depth[1:-1, 1:-1] == _delta.depth)
        assert np.all(_delta.qs == 0)      # field is cleared
        assert np.all(_delta.Vp_dep_sand == 0)
        assert np.all(_delta.Vp_dep_mud == 0)


class TestRouteAllSandParcels:

    def test_route_sand_parcels(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'Np_sed': 1000,
                                      'f_bedload': 0.6})
        _delta = DeltaModel(input_file=p)

        # mock top-level methods / objects
        _delta.log_info = mock.MagicMock()
        _delta._sr = mock.MagicMock()

        # mock the shared tools start indices
        def _patched_starts(inlet, inlet_weights, num_starts):
            return np.random.randint(0, 5, size=(num_starts,))

        patcher = mock.patch(
            'pyDeltaRCM.shared_tools.get_start_indices',
            new=_patched_starts)
        patcher.start()

        # run the method
        _delta.route_all_sand_parcels()

        # methods called
        assert (_delta._sr.run.call_count == 1)
        assert (_delta.log_info.call_count == 3)

        # stop the patch
        patcher.stop()


class TestRouteAllMudParcels:

    def test_route_mud_parcels(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'Np_sed': 1000,
                                      'f_bedload': 0.6})
        _delta = DeltaModel(input_file=p)

        # mock top-level methods / objects
        _delta.log_info = mock.MagicMock()
        _delta._mr = mock.MagicMock()

        # mock the shared tools start indices
        def _patched_starts(inlet, inlet_weights, num_starts):
            return np.random.randint(0, 5, size=(num_starts,))

        patcher = mock.patch(
            'pyDeltaRCM.shared_tools.get_start_indices',
            new=_patched_starts)
        patcher.start()

        # run the method
        _delta.route_all_mud_parcels()

        # methods called
        assert (_delta._mr.run.call_count == 1)
        assert (_delta.log_info.call_count == 3)

        # stop the patch
        patcher.stop()
