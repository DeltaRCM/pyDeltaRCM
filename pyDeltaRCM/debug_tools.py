
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mpl_toolkits.axes_grid1 as axtk
import abc

from . import shared_tools

# tools for water routing algorithms


class debug_tools(abc.ABC):
    """Debugging tools.

    These tools can be invoked as part of a script during runtime or in a
    Python debugging shell.

    Add ``breakpoint()`` to any line in the code to enter the debugger and use
    these tools interactively at that breakpoint. Note, for Python < 3.7 use
    ``pdb.set_trace()``.

    .. testsetup::
        >>> self = pyDeltaRCM.DeltaModel()

    Examples
    --------

    Within a debugging shell:

    .. doctest::

        >>> self.show_attribute('cell_type')
        >>> self.show_ind([144, 22, 33, 34, 35])
        >>> self.show_ind((12, 14), 'bs')
        >>> self.show_ind([(11, 4), (11, 5)], 'g^')
        >>> plt.show()

    .. plot:: debug_tools/debug_demo.py

    """
    def _get_attribute(self, attribute):
        _attr = getattr(self, attribute)
        if not isinstance(_attr, np.ndarray):
            raise TypeError('Attribute must be a numpy.ndarray, but was:'
                            '%s' % str(type(_attr)))
        _attr_shape = _attr.shape
        if len(_attr_shape) != 2:
            raise ValueError('Attribute "{at}" has shape {shp}, '
                             'but must be two-dimensional.'.format(at=attribute,
                                                                   shp=_attr_shape))
        return _attr

    def show_attribute(self, attribute, **kwargs):
        """Show an attribute over the model domain.

        Show any attribute of the :obj:`~pyDeltaRCM.model.DeltaModel` class
        over the model domain.

        Parameters
        ----------
        attribute : :obj:`str`
            Name of the attribute you want to show.

        ax : :obj:`matplotlib.Axes` object, optional
            Which axes to render attribute into. Uses ``gca()`` if no axis is
            provided.

        grid : :obj:`bool`, optional
            Whether to plot a grid over the domain to demarcate individual
            cells. Default is `True` (show the grid).

        block : :obj:`bool`, optional
            Whether to show the plot automatically. Default is `False` (do not
            show automatically).

        """

        if not isinstance(attribute, str):
            raise TypeError('Expected string for `attribute`, but was %s'
                            % type(attribute))
        _attr = self._get_attribute(attribute)
        plot_domain(_attr, **kwargs)

    def show_ind(self, ind, *args, **kwargs):
        """Show points within the model domain.

        Show the location of points (indices) within the model domain. Can
        show points as tuple ``(x, y)``, flat index ``idx``, list of tuples
        ``[(x1, y1), (x2, y2)]``, or list of flat indices ``[idx1, idx2]``.
        Method takes arbitrary `matplotlib` arguments to `plot` the points as
        Matlab-style args (``'r*'``) or keyword arguments (``marker='s'``).

        Parameters
        ----------
        ind : :obj:`tuple`, `int`, `list` of `tuple`, `list` of `int`, `list` of `int` and `tuple`
            Index (indices if list), to plot

        ax : :obj:`matplotlib.Axes` object, optional
            Which axes to render point into. Uses ``gca()`` if no axis is
            provided.

        block : :obj:`bool`, optional
            Whether to show the plot automatically. Default is `False` (do not
            show automatically).

        Other Parameters
        ----------------
        *args : :obj:`str`, optional
            Matlab-style point specifications, e.g., ``'r*'``, ``'bs'``.

        **kwargs : optional
            Any `kwargs` supported by `matplotlib.pyplot.plt`.

        """
        _shape = self.depth.shape
        if isinstance(ind, list) or isinstance(ind, np.ndarray):
            for _, iind in enumerate(ind):
                plot_ind(iind, shape=_shape, *args, **kwargs)
        else:
            plot_ind(ind, shape=_shape, *args, **kwargs)


def plot_domain(attr, ax=None, grid=True, block=False, label=None):
    """Plot the model domain.

    Public function to plot *any* 2d grid with helper display utils.
    This function is called by the DeltaModel method :obj:`show_attribute`.

    Parameters
    ----------
    attr : :obj:`ndarray`
        The data to display, as a 2D numpy array.

    ax : :obj:`matplotlib.Axes`, optional
        Which axes to render point into. Uses ``gca()`` if no axis is
            provided.

    grid : :obj:`bool`, optional
        Whether to plot a grid over the domain to demarcate individual
        cells. Default is `True` (show the grid).

    block : :obj:`bool`, optional
        Whether to show the plot automatically. Default is `False` (do not
        show automatically).

    label : :obj:`str`, optional
        A string describing the field being plotted. If given, will be
        appeneded to the colorbar of the domain plot.
    """
    attr_shape = attr.shape
    if not ax:
        ax = plt.gca()

    cobj = ax.imshow(attr,
                     cmap=plt.get_cmap('viridis'),
                     interpolation='none')
    divider = axtk.axes_divider.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(cobj, cax=cax)
    ax.autoscale(False)
    plt.sca(ax)

    if grid:
        shp = attr_shape
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(np.arange(-.5, shp[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, shp[0], 1), minor=True)
        ax.tick_params(which='minor', length=0)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)

    if label:
        cbar.ax.set_ylabel(label)

    if block:
        plt.show()


def plot_ind(_ind, *args, shape=None, **kwargs):
    """Plot points within the model domain.

    Private method called by :obj:`show_ind`.
    """
    ax = kwargs.pop('ax', None)
    block = kwargs.pop('block', False)

    _shape = shape

    if not ax:
        ax = plt.gca()

    if len(args) == 0:
        args = 'r.',
    if isinstance(_ind, tuple):
        if not len(_ind) == 2:
            raise ValueError('Expected tuple length to be 2, but was %s'
                             % str(len(_ind)))
    else:
        if (_shape is None):
            raise ValueError('Shape of array must be given to unravel index.')
        if isinstance(_ind, np.ndarray):
            _ind = _ind[0]
        _ind = shared_tools.custom_unravel(_ind, _shape)
    plt.plot(_ind[1], _ind[0], *args, **kwargs)

    if block:
        plt.show()
