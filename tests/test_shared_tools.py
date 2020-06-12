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


def test_sand_partition(test_DeltaModel):
    """
    Test for function shared_tools.partition_sand
    """
    nx, ny, qsn = shared_tools.partition_sand(
        test_DeltaModel.qs, 1, 4, 4, 1, 0, 1
    )
    assert nx == 5
    assert ny == 4
    assert qsn[4, 4] == 1
    assert qsn[5, 4] == 1
    assert np.all(qsn[qsn != 1] == 0)


def test_get_steps():
    """
    Test for function shared_tools.get_steps
    """
    new_cells = np.arange(9)
    iwalk = shared_tools.get_iwalk()
    jwalk = shared_tools.get_jwalk()

    d, i, j, a = shared_tools.get_steps(new_cells, iwalk.flatten(), jwalk.flatten())

    d_exp = np.array([1.41421356, 1., 1.41421356, 1., 0.,
             1., 1.41421356, 1., 1.41421356])
    i_exp = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1])
    j_exp = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1])

    assert np.all(np.delete(a, 4))
    assert ~a[4]
    assert i == pytest.approx(i_exp)
    assert j == pytest.approx(j_exp)
    assert d == pytest.approx(d_exp)


def test_update_dirQfield(test_DeltaModel):
    """
    Test for function shared_tools.update_dirQfield
    """
    np.random.seed(test_DeltaModel.seed)
    qx = np.random.uniform(0, 10, 9)
    d = np.array([1, np.sqrt(2), 0])
    astep = np.array([True, True, False])
    inds = np.array([3, 4, 5])
    stepdir = np.array([1, 1, 0])
    qxn = shared_tools.update_dirQfield(np.copy(qx), d, inds, astep, stepdir)
    qxdiff = qxn - qx
    qxdiff_exp = np.array([1, np.sqrt(2) / 2, 0])
    assert np.all(qxdiff[3:6] == pytest.approx(qxdiff_exp))


def test_update_absQfield(test_DeltaModel):
    """
    Test for function shared_tools.update_absQfield
    """
    np.random.seed(test_DeltaModel.seed)
    qw = np.random.uniform(0, 10, 9)
    d = np.array([1, np.sqrt(2), 0])
    astep = np.array([True, True, False])
    inds = np.array([3, 4, 5])
    qwn = shared_tools.update_absQfield(np.copy(qw), d, inds, astep, test_DeltaModel.Qp_water, test_DeltaModel.dx)
    qwdiff = qwn - qw
    diffelem = test_DeltaModel.Qp_water / test_DeltaModel.dx / 2
    qwdiff_exp = np.array([diffelem, diffelem, 0])
    assert np.all(qwdiff[3:6] == pytest.approx(qwdiff_exp))


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


def test_check_for_loops():

    idxs = np.array(
        [[0, 11, 12, 13, 23, 22, 12],
         [0, 1, 2, 3, 4, 5, 16]])
    nidx = np.array([21, 6])
    itt = 6
    free = np.array([1, 1])
    CTR = 4
    L0 = 1
    looped = np.array([0, 0])

    nidx, looped, free = shared_tools.check_for_loops(
        idxs, nidx, itt, L0, looped, (10, 10), CTR, free)

    assert np.all(nidx == [41, 6])
    assert np.all(looped == [1, 0])
    assert np.all(free == [-1, 1])


def test_calculate_new_ind():

    cidx = np.array([12, 16, 16])
    ncel = np.array([6, 1, 4])
    iwalk = shared_tools.get_iwalk()
    jwalk = shared_tools.get_jwalk()

    nidx = shared_tools.calculate_new_ind(cidx, ncel, iwalk.flatten(), jwalk.flatten(), (10, 10))

    nidx_exp = np.array([21, 6, 0])
    assert np.all(nidx == nidx_exp)


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

    wts = shared_tools.get_weight_at_cell(ind, stage, depth, celltype, stage, qx, qy,
                           ivec, jvec, dists, dry_thresh, gamma, theta)
    assert np.all(wts[[0, 1, 2, 5]] == 0)
    assert wts[4] == 0
    assert np.any(wts[[3, 6, 7, 8]] != 0)


def test_version_is_valid():
    v = shared_tools._get_version()
    assert type(v) is str
    dots = [i for i, c in enumerate(v) if c == '.']
    assert len(dots) == 2


@pytest.mark.xfail(raises=IndexError, strict=True)
def test_limit_inds_error(tmp_path):
    """IndexError on corner.

    This test throws an error by trying to index cell 1800 of a 30x60 array.
    This exceeds the limit of the array. I suspect this is a bug with the
    unravel in shared_tools.

    The xfail should be removed when the bug is fixed.
    """

    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 42)
    utilities.write_parameter_to_file(f, 'Length', 30.)
    utilities.write_parameter_to_file(f, 'Width', 60.)
    utilities.write_parameter_to_file(f, 'dx', 1)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.65)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 2):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 4]
    print(delta.eta[:5, 2])

    _exp = np.array([1.7, 0.83358884, -0.9256229,  -1., -1.])
    assert np.all(delta.eta[:5, 2] == pytest.approx(_exp))    


"""
This test cannot be enabled because it causes a segfault. For some reason,
this configuration results in an index error (like the below test) if you run
the config as a normal model run, but produces a segfault inside the test.

@pytest.mark.xfail(raises=IndexError, strict=True)
def test_limits_inds_error_segfault_error(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 43)
    utilities.write_parameter_to_file(f, 'Length', 30.)
    utilities.write_parameter_to_file(f, 'Width', 60.)
    utilities.write_parameter_to_file(f, 'dx', 1)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.65)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 2):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 4]
    print(delta.eta[:5, 2])

    _exp = np.array([1.7, 0.83358884, -0.9256229,  -1., -1.])
    assert np.all(delta.eta[:5, 2] == pytest.approx(_exp))    


@pytest.mark.xfail(raises=IndexError, strict=True)
def test_limit_inds_error(tmp_path):
    # IndexError on corner.

    # This test throws an error by trying to index cell 1800 of a 30x60 array.
    # This exceeds the limit of the array. I suspect this is a bug with the
    # unravel in shared_tools.

    # The xfail should be removed when the bug is fixed.

    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 42)
    utilities.write_parameter_to_file(f, 'Length', 20.)
    utilities.write_parameter_to_file(f, 'Width', 10.)
    utilities.write_parameter_to_file(f, 'dx', 2)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.65)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 2):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 4]
    print(delta.eta[:5, 2])

    _exp = np.array([1.7, 0.83358884, -0.9256229,  -1., -1.])
    assert np.all(delta.eta[:5, 2] == pytest.approx(_exp))    

"""