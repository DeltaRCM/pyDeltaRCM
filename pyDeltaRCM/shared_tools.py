
import numpy as np

from numba import njit, jit, typed

# tools shared between deltaRCM water and sediment routing


def get_iwalk():
    return np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])


def get_jwalk():
    return np.array([[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]])


@njit
def set_random_seed(_seed):
    np.random.seed(_seed)


@njit
def get_random_uniform(N):
    return np.random.uniform(0, 1)


@njit
def get_start_indices(inlet, inlet_weights, num_starts):
    norm_weights = inlet_weights / np.sum(inlet_weights)
    idxs = []
    for x in np.arange(num_starts):
        idxs.append(random_pick(norm_weights))
    idxs = np.array(idxs)
    return inlet.take(idxs)


@njit
def partition_sand(qs, depoPart, py, px, dist, istep, jstep):
    """Spread sand between two cells."""
    if dist > 0:
        # deposition in current cell
        qs[px, py] += depoPart

    px = px + jstep
    py = py + istep

    if dist > 0:
        # deposition in downstream cell
        qs[px, py] += depoPart
    return px, py, qs


@njit
def get_steps(new_cells, iwalk, jwalk):
    """Find the values giving the next step."""
    istep = iwalk[new_cells]
    jstep = jwalk[new_cells]
    dist = np.sqrt(istep * istep + jstep * jstep)

    astep = dist != 0

    return dist, istep, jstep, astep


@njit
def update_dirQfield(qfield, dist, inds, astep, dirstep):
    """Update unit vector of water flux in x or y."""
    for i, ii in enumerate(inds):
        if astep[i]:
            qfield[ii] += dirstep[i] / dist[i]
    return qfield


@njit
def update_absQfield(qfield, dist, inds, astep, Qp_water, dx):
    """Update norm of water flux vector."""
    for i, ii in enumerate(inds):
        if astep[i]:
            qfield[ii] += Qp_water / dx / 2
    return qfield


@njit
def random_pick(prob):
    """Pick number from weighted array.

    Randomly pick a number weighted by array probabilities (len 9)
    Return the index of the selected weight in array probs
    Takes a numpy array that is the precalculated cumulative probability
    around the cell flattened to 1D.
    """
    arr = np.arange(len(prob))
    return arr[np.searchsorted(np.cumsum(prob), get_random_uniform(1))]


@njit
def custom_unravel(i, shape):
    """Unravel indexes for 2D array."""
    if i > (shape[1] * shape[0]):
        raise IndexError("Index is out of matrix bounds")
    x = i // shape[1]
    y = i % shape[1]
    return x, y


@njit
def custom_ravel(tup, shape):
    """Ravel indexes for 2D array."""
    if tup[0] > shape[0] or tup[1] > shape[1]:
        raise IndexError("Index is out of matrix bounds")
    x = tup[0] * shape[1]
    y = tup[1]
    return x + y


@njit
def get_weight_sfc_int(stage, stage_nbrs, qx, qy, ivec, jvec, distances):
    """Determine random walk weight surfaces.

    Determines the surfaces for weighting the random walks based on the stage
    and discharge fields.
    """
    weight_sfc = np.maximum(0, (stage - stage_nbrs) / distances)
    weight_int = np.maximum(0, (qx * jvec + qy * ivec) / distances)
    return weight_sfc, weight_int


@njit
def get_weight_at_cell(ind, weight_sfc, weight_int, depth_nbrs, ct_nbrs,
                       dry_depth, gamma, theta):

    if ind[0] == 0:
        weight_sfc[:3] = np.nan
        weight_int[:3] = np.nan

    drywall = (depth_nbrs <= dry_depth) | (ct_nbrs == -2)
    weight_sfc[drywall] = np.nan
    weight_int[drywall] = np.nan

    if np.nansum(weight_sfc) > 0:
        weight_sfc = weight_sfc / np.nansum(weight_sfc)

    if np.nansum(weight_int) > 0:
        weight_int = weight_int / np.nansum(weight_int)

    weight = gamma * weight_sfc + (1 - gamma) * weight_int
    weight = depth_nbrs ** theta * weight
    weight[depth_nbrs <= dry_depth] = 0

    nanWeight = np.isnan(weight)

    if np.any(weight[~nanWeight] != 0):
        weight = weight / np.nansum(weight)
        weight[nanWeight] = 0
    else:
        weight[~nanWeight] = 1 / np.maximum(1, len(weight[~nanWeight]))
        weight[nanWeight] = 0
    return weight


def _get_version():
    """Extract version from file.

    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()
