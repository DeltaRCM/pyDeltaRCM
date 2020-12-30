# unit tests for shared_tools.py

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM import shared_tools
import utilities
from utilities import test_DeltaModel


def test_set_random_assignments(test_DeltaModel):
    """
    Test for function shared_tools.get_random_uniform and
    test for function shared_tools.set_random_seed
    """
    shared_tools.set_random_seed(test_DeltaModel.seed)
    got = shared_tools.get_random_uniform(1)
    _exp = 0.5488135039273248
    assert got == pytest.approx(_exp)


def test_get_steps():
    """
    Test for function shared_tools.get_steps
    """
    new_cells = np.arange(9)
    iwalk = shared_tools.get_iwalk()
    jwalk = shared_tools.get_jwalk()

    d, i, j, a = shared_tools.get_steps(
        new_cells, iwalk.flatten(), jwalk.flatten())

    d_exp = np.array([1.41421356, 1., 1.41421356, 1., 0.,
                      1., 1.41421356, 1., 1.41421356])
    i_exp = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1])
    j_exp = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1])

    assert np.all(np.delete(a, 4))
    assert ~a[4]
    assert i == pytest.approx(i_exp)
    assert j == pytest.approx(j_exp)
    assert d == pytest.approx(d_exp)


def test_random_pick():
    """
    Test for function shared_tools.random_pick
    """
    # define probs array of zeros with a single 1 value
    probs = np.zeros((8,))
    probs[0] = 1
    # should return first index
    assert shared_tools.random_pick(probs) == 0


def test_random_pick_anybut_first(test_DeltaModel):
    """
    Test for function shared_tools.random_pick
    """
    shared_tools.set_random_seed(test_DeltaModel.seed)
    probs = (1 / 7) * np.ones((3, 3), dtype=np.float64)
    probs[0, 0] = 0
    probs[1, 1] = 0
    # should never return first index
    _rets = np.zeros((100,))
    for i in range(100):
        _rets[i] = shared_tools.random_pick(probs.flatten())
    assert np.all(_rets != 0)
    assert np.all(_rets != 4)  # THIS LINE NEEDS TO PASS!!
    assert np.sum(_rets == 1) > 0
    assert np.sum(_rets == 2) > 0
    assert np.sum(_rets == 3) > 0
    assert np.sum(_rets == 5) > 0
    assert np.sum(_rets == 6) > 0
    assert np.sum(_rets == 7) > 0
    assert np.sum(_rets == 8) > 0


@pytest.mark.xfail(strict=True, reason='cannot mock a jitted function')
def test_random_pick_edgecase(mocker):
    """
    This test PASSES if `export NUMBA_DISABLE_JIT=1` is declared.
    I.e., if the mock can work...
    """
    mocked = mocker.patch('pyDeltaRCM.shared_tools.get_random_uniform')

    probs = np.array([0, 0,          0.23647638,
                      0, 0,          0.49098423,
                      0, 0.01803146, 0.2545079])
    mocked.return_value = 0
    assert shared_tools.random_pick(probs.flatten()) == 0
    mocked.return_value = 0.1
    assert shared_tools.random_pick(probs.flatten()) == 2
    mocked.return_value = 0.9999
    assert shared_tools.random_pick(probs.flatten()) == 8
    mocked.return_value = 0.9999999
    assert shared_tools.random_pick(probs.flatten()) == 8
    mocked.return_value = 1
    assert shared_tools.random_pick(probs.flatten()) == 8


def test_custom_unravel_square():
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


def test_custom_unravel_rectangle():
    arr = np.arange(50).reshape((5, 10))
    # test a few spots
    x, y = shared_tools.custom_unravel(19, arr.shape)
    assert x == 1
    assert y == 9
    x, y = shared_tools.custom_unravel(34, arr.shape)
    assert x == 3
    assert y == 4


def test_custom_unravel_exceed_error():
    arr = np.arange(9).reshape((3, 3))
    # next line should throw IndexError
    with pytest.raises(IndexError):
        x, y = shared_tools.custom_unravel(99, arr.shape)


def test_get_weight_sfc_int(test_DeltaModel):

    np.random.seed(test_DeltaModel.seed)
    stage = np.random.uniform(0.5, 1, 9)
    qx = 1
    qy = 1
    ivec = test_DeltaModel.ivec_flat
    jvec = test_DeltaModel.jvec_flat
    dists = test_DeltaModel.distances_flat

    weight_sfc, weight_int = shared_tools.get_weight_sfc_int(stage, stage,
                                                             qx, qy,
                                                             ivec, jvec,
                                                             dists)
    assert np.all(weight_sfc == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert np.all(weight_int == np.array([0, 0, 0, 0, 0, 1, 0, 1, 1]))


def test_get_weight_at_cell(test_DeltaModel):

    ind = (0, 4)
    np.random.seed(test_DeltaModel.seed)
    stage = np.random.uniform(0.5, 1, 9)
    eta = np.random.uniform(0, 0.85, 9)
    depth = stage - eta
    depth[depth < 0] = 0
    celltype = np.array([-2, -2, -2, 1, 1, -2, 0, 0, 0])
    qx = 1
    qy = 1
    ivec = test_DeltaModel.ivec.flatten()
    jvec = test_DeltaModel.jvec.flatten()
    dists = test_DeltaModel.distances.flatten()
    dry_thresh = 0.1
    gamma = 0.001962
    theta = 1
    weight_sfc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    weight_int = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1], dtype=np.float64)

    wts = shared_tools.get_weight_at_cell(ind, weight_sfc, weight_int, depth,
                                          celltype, dry_thresh, gamma, theta)
    assert np.all(wts[[0, 1, 2, 5]] == 0)
    assert wts[4] == 0
    assert np.any(wts[[3, 6, 7, 8]] != 0)


def test_version_is_valid():
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
