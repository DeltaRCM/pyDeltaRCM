
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
def check_for_loops(indices, inds, it, L0, loopedout, domain_shape, CTR, free_surf_flag):

    looped = typed.List()  # numba typed list for iteration
    for i in np.arange(len(indices)):
        row = indices[i, :]
        v = len(row[row > 0]) != len(set(row[row > 0]))
        looped.append(v)
    travel = (0, it)

    for n in range(indices.shape[0]):
        ind = inds[n]
        if looped[n] and (ind > 0) and (max(travel) > L0):
            loopedout[n] += 1
            px, py = custom_unravel(ind, domain_shape)
            Fx = px - 1
            Fy = py - CTR

            Fw = np.sqrt(Fx**2 + Fy**2)

            if Fw != 0:
                px = px + np.round(Fx / Fw * 5.)
                py = py + np.round(Fy / Fw * 5.)

            px = max(px, L0)
            px = int(min(domain_shape[0] - 2, px))

            py = max(1, py)
            py = int(min(domain_shape[1] - 2, py))

            nind = custom_ravel((px, py), domain_shape)

            inds[n] = nind

            free_surf_flag[n] = -1

    return inds, loopedout, free_surf_flag


@njit
def calculate_new_ind(indices, new_cells, iwalk, jwalk, domain_shape):
    newbies = []
    for p, q in zip(indices, new_cells):
        if q != 4:
            ind_tuple = custom_unravel(p, domain_shape)
            new_ind = (ind_tuple[0] + jwalk[q],
                       ind_tuple[1] + iwalk[q])
            newbies.append(custom_ravel(new_ind, domain_shape))
        else:
            newbies.append(0)

    return np.array(newbies)


@njit
def get_weight_at_cell(ind, stage_nbrs, depth_nbrs, ct_nbrs, stage, qx, qy,
                       ivec, jvec, distances, dry_depth, gamma, theta):

    weight_sfc = np.maximum(0, (stage - stage_nbrs) / distances)

    weight_int = np.maximum(0, (qx * jvec + qy * ivec) / distances)

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
        weight[~nanWeight] = 1 / len(weight[~nanWeight])
        weight[nanWeight] = 0
    return weight


def _get_version():
    """Extract version from file.

    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()
