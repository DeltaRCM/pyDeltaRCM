
import numpy as np
import yaml
import re
import contextlib
import tempfile
import os

from numba import njit, _helperlib

# tools shared between deltaRCM water and sediment routing

# useful constants
sec_in_day = 86400
day_in_yr = 365.25
earth_grav = 9.81


@njit
def set_random_seed(_seed):
    """Set the random seed from an integer.

    Set the seed of the random number generator with a single integer.
    Importantly, this function will affect the state of the `numba` random
    number generator, rather than the `numpy` generator.

    This function is called during model intialization, but is **not** called
    during loading a model from checkpoint
    (:meth:`~pyDeltaRCM.init_tools.init_tools.load_checkpoint`) where the
    random number generator state is configured via the
    :func:`set_random_state` function.

    .. important::

        This function is preferred to setting the random seed directly. If you
        are working with model subclassing, interacting with the `numpy`
        random number generator directly will lead to a model that is **not
        reproducible!**

    Internally, the method is simply the below line; however it is *crucial*
    this happens inside a jitted function, in order to affect the state of the
    `numba` random number generator seed.

    .. code::

        np.random.seed(_seed)

    Parameters
    ----------
    _seed : :obj:`int`
        Integer to use as the random number generator seed. Should be in
        interval ``[0, 2^32]``.
    """
    np.random.seed(_seed)


def get_random_state():
    """Get the random state as a tuple.

    Get the random state from a tuple in the form returned by
    :func:`numba._helperlib.rnd_get_state`. This tuple contains the
    necessary information for resuming a checkpoint from the exact same random
    number generator state.

    .. important::

        You probably do not need to use this function. Be sure you know
        *exactly* how you are affecting the random number generator state
        before using this function, or you are likely to create a model that
        cannot be peproduced.

    See also :func:`set_random_seed`.

    Returns
    -------
    state : :obj:`tuple`
        Random number generator state as a tuple.
    """
    ptr = _helperlib.rnd_get_np_state_ptr()
    return _helperlib.rnd_get_state(ptr)


def set_random_state(_state_tuple):
    """Set the random state from a tuple.

    Set the random state from a tuple in the form returned by
    :func:`numba._helperlib.rnd_get_state`.

    .. important::

        You probably do not need to use this function. Be sure you know
        *exactly* how you are affecting the random number generator state
        before using this function, or you are likely to create a model that
        cannot be peproduced.

    See also :func:`set_random_seed`.

    Parameters
    ----------
    state : :obj:`tuple`
        Random number generator state as a tuple.
    """
    ptr = _helperlib.rnd_get_np_state_ptr()
    _helperlib.rnd_set_state(ptr, _state_tuple)


@njit
def get_random_uniform(limit):
    """Get a random number from the uniform distribution.

    Get a random number from the uniform distribution, defined over the
    interval ``[0, limit]``, where `limit` is an input parameter (usually
    ``1``).

    .. hint::

        Function returns only a single float. Wrap the call in a simple for
        loop to get many random numbers.

    Parameters
    ----------
    limit : :obj:`float`
        Upper limit to distribution to draw from.

    Returns
    -------
    :obj:`float`
        Float value from the random number generator in the interval defined.

    Examples
    --------
    .. testsetup::

        from pyDeltaRCM.shared_tools import get_random_uniform, set_random_seed
        set_random_seed(0)

    .. doctest::

        >>> get_random_uniform(1)
        0.5488135039273248

        >>> get_random_uniform(0.1)
        0.07151893663724194

    """
    return np.random.uniform(0, limit)


@njit
def get_start_indices(inlet, inlet_weights, num_starts):
    """Get start indices.

    Reutrn a randomly generated list of starting points for parcel routing.
    These starting points are selected from the `inlet` array.

    Parameters
    ----------
    inlet : ndarray
        Array of inlet cells.

    inlet_weights : ndarray
        Array of weights to select items from :obj:`inlet`. Should have same
        dimensions.

    num_starts : int
        Number of starting points to generate.

    Returns
    -------
    start_indices : ndarray
        :obj:`num_starts` starting points, generated from :obj:`inlet`
         according to weights in :obj:`inlet_weights`. 
    """
    norm_weights = inlet_weights / np.sum(inlet_weights)
    idxs = []
    for _ in np.arange(num_starts):
        idxs.append(random_pick(norm_weights))
    idxs = np.array(idxs)
    return inlet.take(idxs)


@njit
def get_steps(new_direction, iwalk, jwalk):
    """Find the values given the next step.

    Get the steps for updating discharge and velocity arrays based on the
    direction of each step.
    """
    istep = iwalk[new_direction]
    jstep = jwalk[new_direction]
    dist = np.sqrt(istep * istep + jstep * jstep)
    astep = (dist != 0)

    return dist, istep, jstep, astep


@njit
def random_pick(prob):
    """Pick number from weighted array.

    Randomly pick a number weighted by array probabilities (len 9)
    Return the index of the selected weight in array probs
    Takes a numpy array that is the precalculated probability
    around the cell flattened to 1D.

    Examples
    --------

    .. todo:: Examples needed (pull from tests?).

    """
    arr = np.arange(len(prob))
    cumprob = np.cumsum(prob)
    return arr[np.searchsorted(cumprob, get_random_uniform(cumprob[-1]))]


@njit('UniTuple(int64, 2)(int64, UniTuple(int64, 2))')
def custom_unravel(i, shape):
    """Unravel indexes for 2D array.

    This is a jitted function, equivalent to the `numpy` implementation of
    `unravel_index`, but configured to accept only a single integer as
    the index to unravel.

    Parameters
    ----------
    i : :obj:`int`
        Integer index to unravel.

    shape : :obj:`tuple` of :obj:`int`
        Two element tuple of integers indicating the shape of the
        two-dimensional array from which :obj:`i` will be unravelled.

    Returns
    -------
    x : :obj:`int`
        `x` index of :obj:`i` into array of shape :obj:`shape`.

    y : :obj:`int`
        `y` index of :obj:`i` into array of shape :obj:`shape`.

    Examples
    --------
    .. testsetup::

        from pyDeltaRCM.shared_tools import custom_unravel

    .. doctest::

        >>> _shape = (100, 200)  # e.g., delta.eta.shape

        >>> np.unravel_index(5124, _shape)
        (25, 124)

        >>> custom_unravel(5124, _shape)
        (25, 124)
    """
    if i > (shape[1] * shape[0]):
        raise IndexError("Index is out of array bounds")
    x = i // shape[1]
    y = i % shape[1]
    return x, y


@njit('int64(UniTuple(int64, 2), UniTuple(int64, 2))')
def custom_ravel(tup, shape):
    """Ravel indexes for 2D array.

    This is a jitted function, equivalent to the `numpy` implementation of
    `ravel_multi_index`, but configured to accept only a single tuple as
    the index to ravel, and only works for two-dimensional array.

    Parameters
    ----------
    tup : :obj:`tuple` of :obj:`int`
        Two element `tuple` with the `x` and `y` index into an array of
        shape :obj:`shape` to ravel.

    shape : :obj:`tuple` of :obj:`int`
        Two element tuple of integers indicating the shape of the
        two-dimensional array from which :obj:`tup` will be ravelled.

    Returns
    -------
    i : :obj:`int`
        Ravelled integer index into array of shape :obj:`shape`.

    Examples
    --------
    .. testsetup::

        from pyDeltaRCM.shared_tools import custom_ravel

    .. doctest::

        >>> _shape = (100, 200)  # e.g., delta.eta.shape

        >>> np.ravel_multi_index((25, 124), _shape)
        5124

        >>> custom_ravel((25, 124), _shape)
        5124
    """
    if tup[0] > shape[0] or tup[1] > shape[1]:
        raise IndexError("Index is out of array bounds")
    x = tup[0] * shape[1]
    y = tup[1]
    return x + y


@njit
def custom_pad(arr):
    """Pad an array.

    This is a jitted function, equivalent to the `numpy` implementation of
    `pad` with certain optional parameters:

    .. code::

        np.pad(arr, 1, 'edge')

    In pyDeltaRCM model fields (e.g., `depth`) are frequently padded to enable
    straightforward slicing of the neighbors of a given cell; i.e., padding
    guarantees that all cells will have 9 neighbors.

    Parameters
    ----------
    arr : :obj:`ndarray`
        Array to pad.

    Returns
    -------
    pad : :obj:`ndarray`
        Padded array.

    Examples
    --------
    .. testsetup::

        from pyDeltaRCM.shared_tools import custom_pad

    Consider a model domain of size `(4, 8)`

    .. doctest::

        >>> arr = np.arange(32).reshape(4, 8)

        >>> np_pad = np.pad(arr, 1, 'edge')
        >>> cust_pad = custom_pad(arr)
        >>> np.all(np_pad == cust_pad)
        True
        >>> cust_pad.shape
        (6, 10)

    which enables all elements of the original `(4, 8)` array to be safely
    sliced:

    .. doctest::

        >>> for i in range(4):
        ...     for j in range(8):
        ...         slc = cust_pad[i:i+3, j:j+3]

        >>> slc.shape  # will always be a (3, 3) 9-cell neighborhood
        (3, 3)

    """
    old_shape = arr.shape
    new_shape = (old_shape[0]+2, old_shape[1]+2)
    pad = np.zeros(new_shape, dtype=arr.dtype)

    # center
    pad[1:-1, 1:-1] = arr

    # edges
    pad[1:-1, 0] = arr[:, 0]  # left
    pad[1:-1, -1] = arr[:, -1]  # right
    pad[0, 1:-1] = arr[0, :]  # top
    pad[-1, 1:-1] = arr[-1, :]  # bottom

    # corners
    pad[0, 0] = arr[0, 0]  # ul
    pad[0, -1] = arr[0, -1]  # ur
    pad[-1, 0] = arr[-1, 0]  # ll
    pad[-1, -1] = arr[-1, -1]  # lr

    return pad


@njit
def get_weight_sfc_int(stage, stage_nbrs, qx, qy, ivec, jvec, distances):
    """Determine random walk weight surfaces.

    Determines the surfaces for weighting the random walks based on the stage
    and discharge fields.

    .. todo::

        Expand description. Where is it used? Include an example? Equations?
        Link to Hydrodyanmics doc?
    """
    weight_sfc = np.maximum(0, (stage - stage_nbrs) / distances)
    weight_int = np.maximum(0, (qx * jvec + qy * ivec) / distances)
    return weight_sfc, weight_int


def _get_version():
    """Extract version from file.

    Extract version number from single file, and make it availabe everywhere.
    This function is used to make sure that when the version number is
    incremented in the `pyDeltaRCM._version` module, it is properly updated
    throughout documentation and version-controlled release.

    .. note:: You probably don't need to use this function.

    Returns
    -------
    ver_str : :obj:`str`
        Version number, as a string.

    Examples
    --------
    .. doctest::

        >>> pyDeltaRCM.shared_tools._get_version()  # doctest: +SKIP
    """
    from . import _version
    return _version.__version__()


@contextlib.contextmanager
def _docs_temp_directory():
    """Helper for creating and tearing down models in documentation.

    This function should be used as a context manager, to create a DeltaModel
    with a temporary folder as output, rather than anywhere in the project
    structure.

    Examples
    --------
    .. testsetup::

        from pyDeltaRCM.shared_tools import _docs_temp_directory

    .. doctest::

        >>> with _docs_temp_directory() as output_dir:
        ...     delta = pyDeltaRCM.DeltaModel(out_dir=output_dir)
    """
    tmpdir = tempfile.TemporaryDirectory()
    output_path = os.path.join(tmpdir.name, 'output')
    yield output_path
    tmpdir.cleanup()


def custom_yaml_loader():
    """A custom YAML loader to handle scientific notation.

    We are waiting for upstream fix here:
        https://github.com/yaml/pyyaml/pull/174

    Returns
    -------
    loader : :obj:`pyyaml.Loader` with custom resolver

    Examples
    --------
    The custom loader can be used as:

    .. code::

        >>> loader = pyDeltaRCM.shared_tools.custom_yaml_loader()

        >>> a_file = open('/path/to/file.yaml', mode='r')
        >>> yaml_as_dict = yaml.load(a_file, Loader=loader)
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                       |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                       |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                       |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                       |[-+]?\.(?:inf|Inf|INF)
                       |\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return loader


def scale_model_time(time, If=1, units='seconds'):
    """Scale the model time to "real" time.

    Model time is executed as assumed flooding conditions, and executed at the
    per-second level, with a multi-second timestep. This model design
    implicitly assumes that the delta is *always* receiving a large volume of
    sediment and water at the inlet. This is unrealistic, given that rivers
    flood only during a small portion of the year, and this is when
    morphodynamic activity is largest. See :doc:`../../info/modeltime` for a
    complete description of this assumption, and how to work with the
    assumption in configuring the model.

    Using this assumption, it is possible to scale up model time to "real"
    time, by assuming an *intermittency factor*. This intermittency factor is
    the fraction of unit-time that a river is assumed to be flooding.

    .. math::

        t_r = \dfrac{t}{I_f \cdot S_f}

    where :math:`t` is the model time (:obj:`~pyDeltaRCM.DeltaModel.time`),
    :math:`t_r` is the "real" scaled time, :math:`I_f` is the
    intermittency factor, and :math:`S_f` is the scale factor to convert base
    units of seconds to units specified as an input argument. Note that this
    function uses :obj:`_scale_factor` internally for this conversion.

    Parameters
    ----------
    time : :obj:`float`
        The model time, in seconds.

    If : :obj:`float`, optional
        Intermittency factor, fraction of time represented by morphodynamic
        activity. Should be in interval (0, 1]. Defaults to 1 if not provided,
        i.e., no scaling is performed.

    units : :obj:`str`, optional
        The units to convert the scaled time to. Default is to return the
        scaled time in seconds (`seconds`), but optionally supply argument
        `days` or `years` for unit conversion.

    Returns
    -------
    scaled : :obj:`float`
        Scaled time, in :obj:`units`, assuming the intermittency factor
        :obj:`If`.

    Raises
    ------
    ValueError
        if the value for intermittency is not ``0 < If <= 1``.
    """
    if (If <= 0) or (If > 1):
        raise ValueError(
            'Intermittency `If` is not 0 < If <= 1: %s' % str(If))

    return time / _scale_factor(If, units)


def _scale_factor(If, units):
    """Scaling factor between model time and "real" time.

    The scaling factor relates the model time to a real worl time, by the
    assumed intermittency factor and the user-specified units for output.

    Parameters
    ----------
    If : :obj:`float`
        Intermittency factor, fraction of time represented by morphodynamic
        activity. **Must** be in interval (0, 1].

    units : :obj:`str`
        The units to convert the scaled time to. Must be a string in
        `['seconds', 'days', 'years']`.

    """
    if units == 'seconds':
        S_f = 1
    elif units == 'days':
        S_f = sec_in_day
    elif units == 'years':
        S_f = sec_in_day * day_in_yr
    else:
        raise ValueError('Bad value for `units`: %s' % str(units))
    return (If * S_f)
