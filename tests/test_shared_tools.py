# unit tests for shared_tools.py

import pytest

import numpy as np

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM import shared_tools
from . import utilities


class TestGetAndSetRandom:

    def test_set_random_get_expected(self, tmp_path):
        """
        Test for function shared_tools.get_random_uniform and
        test for function shared_tools.set_random_seed
        """
        # set the random seed
        shared_tools.set_random_seed(0)

        # get back an expected value
        got = shared_tools.get_random_uniform(1)

        # assertion against expected value
        _exp = 0.5488135039273248
        assert got == pytest.approx(_exp)


class TestGetSteps:

    def test_get_steps(self, tmp_path):
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        new_cells = np.arange(9)
        iwalk = delta.iwalk
        jwalk = delta.jwalk

        d, i, j, a = shared_tools.get_steps(
            new_cells, iwalk.flatten(), jwalk.flatten())

        d_exp = np.array([1.41421356, 1., 1.41421356, 1., 0.,
                          1., 1.41421356, 1., 1.41421356])
        i_exp = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1])
        j_exp = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1])

        # assertions
        assert np.all(np.delete(a, 4))
        assert ~a[4]
        assert i == pytest.approx(i_exp)
        assert j == pytest.approx(j_exp)
        assert d == pytest.approx(d_exp)


class TestRandomPick:

    def test_random_pick(self):
        """
        Test for function shared_tools.random_pick
        """
        # define probs array of zeros with a single 1 value
        probs = np.zeros((9,))
        probs[0] = 1
        # should return first index
        assert shared_tools.random_pick(probs) == 0

    def test_random_pick_diff_fixed(self):
        """
        Test for function shared_tools.random_pick
        """
        # define probs array of zeros with a single 1 value
        probs = np.zeros((9,))
        probs[2] = 1
        # should return third index
        assert shared_tools.random_pick(probs) == 2

    def test_random_pick_not_zeroed(self):
        """
        Test for function shared_tools.random_pick
        """
        # define probs array of zeros with a single 1 value
        probs = np.zeros((9,))
        probs[:] = 1/6
        probs[4] = 0
        probs[5] = 0
        probs[8] = 0
        assert np.sum(probs) == 1
        # should return first index
        _rets = np.zeros((100,))
        for i in range(100):
            _rets[i] = shared_tools.random_pick(probs)
        assert np.all(_rets != 4)
        assert np.all(_rets != 5)
        assert np.all(_rets != 8)

    def test_random_pick_distribution(self):
        """
        Test for function shared_tools.random_pick
        """
        # define probs array of zeros with a single 1 value
        probs = np.array([1/6, 1/3, 0,
                          1/12, 1/12, 1/6,
                          0, 0, 1/6])
        assert np.sum(probs) == 1
        _rets = np.zeros((10000,))
        for i in range(10000):
            _rets[i] = shared_tools.random_pick(probs)
        _bins = np.arange(-0.5, 9, step=1.0)
        _hist, _ = np.histogram(_rets, bins=_bins)
        _binedge = _bins[:-1]
        _histnorm = _hist / np.sum(_hist)

        assert np.all(_histnorm == pytest.approx(probs, rel=0.10))

    def test_random_pick_anybut_first(self, tmp_path):
        """
        Test for function shared_tools.random_pick
        """
        probs = (1 / 7) * np.ones((3, 3), dtype=np.float64)
        probs[0, 0] = 0
        probs[1, 1] = 0
        probs_flat = probs.flatten()

        assert np.sum(probs_flat) == 1
        # should never return first index
        _rets = np.zeros((100,))
        for i in range(100):
            _rets[i] = shared_tools.random_pick(probs_flat)

        assert np.all(_rets != 0)
        assert np.all(_rets != 4)  # THIS LINE NEEDS TO PASS!!
        assert np.sum(_rets == 1) > 0
        assert np.sum(_rets == 2) > 0
        assert np.sum(_rets == 3) > 0
        assert np.sum(_rets == 5) > 0
        assert np.sum(_rets == 6) > 0
        assert np.sum(_rets == 7) > 0
        assert np.sum(_rets == 8) > 0


class TestCustomUnravel:

    def test_custom_unravel_square(self):
        arr = np.arange(9).reshape((3, 3))
        # test upper left corner
        x, y = shared_tools.custom_unravel(0, arr.shape)
        assert x == 0
        assert y == 0
        # test center
        x, y = shared_tools.custom_unravel(4, arr.shape)
        assert x == 1
        assert y == 1
        # test off-center
        x, y = shared_tools.custom_unravel(5, arr.shape)
        assert x == 1
        assert y == 2
        # test lower right corner
        x, y = shared_tools.custom_unravel(8, arr.shape)
        assert x == 2
        assert y == 2

    def test_custom_unravel_rectangle(self):
        arr = np.arange(50).reshape((5, 10))
        # test a few spots
        x, y = shared_tools.custom_unravel(19, arr.shape)
        assert x == 1
        assert y == 9
        x, y = shared_tools.custom_unravel(34, arr.shape)
        assert x == 3
        assert y == 4

    def test_custom_unravel_exceed_error(self):
        arr = np.arange(9).reshape((3, 3))
        # next line should throw IndexError
        with pytest.raises(IndexError):
            x, y = shared_tools.custom_unravel(99, arr.shape)


class TestCustomRavel:

    def test_custom_ravel_square(self):
        arr = np.arange(9).reshape((3, 3))
        # test upper left corner
        tup = (0, 0)
        i = shared_tools.custom_ravel(tup, arr.shape)
        assert i == 0
        # test center
        tup = (1, 1)
        i = shared_tools.custom_ravel(tup, arr.shape)
        assert i == 4
        # test off-center
        tup = (1, 2)
        i = shared_tools.custom_ravel(tup, arr.shape)
        assert i == 5
        # test lower right corner
        tup = (2, 2)
        i = shared_tools.custom_ravel(tup, arr.shape)
        assert i == 8

    def test_custom_ravel_rectangle(self):
        arr = np.arange(50).reshape((5, 10))
        # test a few spots
        tup = (1, 9)
        i = shared_tools.custom_ravel(tup, arr.shape)
        assert i == 19
        tup = (3, 4)
        i = shared_tools.custom_ravel(tup, arr.shape)
        assert i == 34

    def test_custom_ravel_exceed_error(self):
        arr = np.arange(9).reshape((3, 3))
        # next line should throw IndexError
        with pytest.raises(IndexError):
            _ = shared_tools.custom_ravel((50, 50), arr.shape)


class TestGetWeightSfcInt:

    def test_get_weight_sfc_int(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        delta = DeltaModel(input_file=p)

        stage = np.random.uniform(0.5, 1, 9)
        qx = 1
        qy = 1
        ivec = delta.ivec_flat
        jvec = delta.jvec_flat
        dists = delta.distances_flat

        weight_sfc, weight_int = shared_tools.get_weight_sfc_int(
            stage[4], stage, qx, qy, ivec, jvec, dists)
        # assert np.all(weight_sfc == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
        # assert np.all(weight_int == np.array([0, 0, 0, 0, 0, 1, 0, 1, 1]))
        assert np.all(weight_sfc <= 1)
        assert np.all(weight_int <= 1)
        assert np.all(weight_sfc >= 0)
        assert np.all(weight_int >= 0)


class TestVerisonSpecifications:

    def test_version_is_valid(self):
        v = shared_tools._get_version()
        assert type(v) is str
        dots = [i for i, c in enumerate(v) if c == '.']
        assert len(dots) == 2


class TestScaleModelTime():

    def test_defaults(self):
        scaled = shared_tools.scale_model_time(86400)
        assert scaled == 86400

    def test_If(self):
        scaled = shared_tools.scale_model_time(86400, 0.1)
        assert scaled == 86400 / 0.1
        scaled = shared_tools.scale_model_time(86400, 0.5)
        assert scaled == 86400 / 0.5
        scaled = shared_tools.scale_model_time(86400, 0.9)
        assert scaled == 86400 / 0.9
        scaled = shared_tools.scale_model_time(86400, 1)
        assert scaled == 86400
        scaled = shared_tools.scale_model_time(86400, 1e-15)
        assert scaled == 86400 / 1e-15
        with pytest.raises(ValueError, match='Intermittency `If` .*'):
            scaled = shared_tools.scale_model_time(86400, 0)
        with pytest.raises(ValueError, match='Intermittency `If` .*'):
            scaled = shared_tools.scale_model_time(86400, 1.01)

    def test_units(self):
        scaled = shared_tools.scale_model_time(86400, units='seconds')
        assert scaled == 86400
        scaled = shared_tools.scale_model_time(86400, units='days')
        assert scaled == 1
        scaled = shared_tools.scale_model_time(86400, units='years')
        assert scaled == (1 / 365.25)
        with pytest.raises(ValueError):
            scaled = shared_tools.scale_model_time(86400, units='badstr')

    def test_combinations(self):
        scaled = shared_tools.scale_model_time(86400, If=0.1, units='days')
        assert scaled == 10
        scaled = shared_tools.scale_model_time(2 * 86400, If=0.1, units='days')
        assert scaled == 20
        scaled = shared_tools.scale_model_time(
            365.25 * 86400, If=1, units='years')
        assert scaled == 1
        scaled = shared_tools.scale_model_time(
            365.25 * 86400, If=0.1, units='years')
        assert scaled == 10
