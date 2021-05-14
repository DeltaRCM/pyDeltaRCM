
import numpy as np
import matplotlib
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

    Examples
    --------

    Within a debugging shell:

    .. code::

        >>> self.show_attribute('cell_type', grid=False)
        >>> self.show_ind([3378, 9145, 11568, 514, 13558])
        >>> self.show_ind((42, 94), 'bs')
        >>> self.show_ind([(41, 8), (42, 10)], 'g^')
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

    def show_line(self, ind, *args, multiline=False, nozeros=False, **kwargs):
        """Show line within the model domain.

        Show the location of lines (a series of connected indices) within the
        model domain. Can show points as a list/array of flat index ``[i0, i1,
        i2]``, list of tuples ``[(x1, y1), (x2, y2)]``, array of pairs ``[[x1,
        y1], [x2, y2]]``, or array of lines if given as flat index ``[[l0i0,
        l1i0, l2i0], [l0i1, l1i1, l2i1]]``. Method takes arbitrary
        `matplotlib` arguments to `plot` the points as Matlab-style args
        (``'r*'``) or keyword arguments (``marker='s'``).

        Parameters
        ----------
        ind : :obj: Nx2 `ndarray`, `list` of `tuple`, `list` of `int`, `ndarray` of `int`
            Indicies to plot.

        ax : :obj:`matplotlib.Axes` object, optional
            Which axes to render point into. Uses ``gca()`` if no axis is
            provided.

        block : :obj:`bool`, optional
            Whether to show the plot automatically. Default is `False` (do not
            show automatically).

        multiline : :obj:`bool`, optional
            When a 2D array is passed as `ind`, this boolean indicates whether
            to treat each column of the array as a separate line. Default is
            `False`.

        nozeros : :obj:`bool`, optional
            Whether to show segements of lines with a flat index `== 0 `.
            Default is `False`.

        Other Parameters
        ----------------
        *args : :obj:`str`, optional
            Matlab-style point specifications, e.g., ``'r*'``, ``'bs'``.

        **kwargs : optional
            Any `kwargs` supported by `matplotlib.pyplot.plt`.

        Returns
        -------
        lines : :obj:`list`
            Lines plotted by call to `show_line`, returned as a list, even if
            only one line plotted.

        """
        _shape = self.depth.shape
        if isinstance(ind, list):
            for _, iind in enumerate(ind):
                _l = plot_line(iind.flatten(),
                               shape=_shape, nozeros=nozeros,
                               *args, **kwargs)
                lines = [_l]
        elif isinstance(ind, np.ndarray):
            if ind.ndim > 2:
                raise NotImplementedError('Not implemented for arrays > 2d.')
            elif ind.ndim > 1:
                if multiline:
                    # travel along axis, extracting lines
                    cm = matplotlib.cm.get_cmap('tab10')
                    lines = []
                    for i in np.arange(ind.shape[1]):
                        _l = plot_line(ind[:, i], *args, multiline=multiline,
                                       nozeros=nozeros, shape=_shape,
                                       color=cm(i), **kwargs)
                        lines.append(_l)
                else:
                    _l = plot_line(ind, *args, nozeros=nozeros, **kwargs)
                    lines = [_l]
            else:
                _l = plot_line(ind.flatten(),
                               shape=_shape, nozeros=nozeros,
                               *args, **kwargs)
                lines = [_l]

        else:
            raise NotImplementedError
        return lines


def plot_domain(attr, ax=None, grid=True, block=False, label=None, **kwargs):
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

    cmap = kwargs.pop('cmap', None)
    if (cmap is None):
        cmap = plt.get_cmap('viridis')
    # else:
        # cmap = plt.get_cmap(cmap)

    cobj = ax.imshow(attr,
                     cmap=cmap,
                     interpolation='none',
                     **kwargs)
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

    .. todo:: write a complete docstring with parameters etc.

    Method called by :obj:`show_ind`.

    Examples
    --------

    .. todo:: add examples, pull from tests.
    """
    ax = kwargs.pop('ax', None)
    block = kwargs.pop('block', False)

    _shape = shape

    if not ax:
        ax = plt.gca()

    if len(args) == 0:
        # args = 'r.',
        if not ('color' in kwargs.keys()):
            kwargs['color'] = 'red'
        if not ('marker' in kwargs.keys()):
            kwargs['marker'] = '.'
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
    ax.plot(_ind[1], _ind[0], *args, **kwargs)

    if block:
        plt.show()


def plot_line(_ind, *args, shape=None, nozeros=False, **kwargs):
    """Plot a line within the model domain.

    Method called by :obj:`show_line`.

    .. todo:: write a complete docstring with parameters etc.

    Examples
    --------

    .. todo:: add examples, pull from tests.
    """
    ax = kwargs.pop('ax', None)
    block = kwargs.pop('block', False)
    multiline = kwargs.pop('multiline', False)

    _shape = shape

    if not ax:
        ax = plt.gca()

    if len(args) == 0:
        # args = 'k-',
        if not ('color' in kwargs.keys()):
            kwargs['color'] = 'k'
        if not ('marker' in kwargs.keys()):
            kwargs['ls'] = '-'
    if isinstance(_ind, tuple):
        raise NotImplementedError
        # if not len(_ind) == 2:
        #     raise ValueError('Expected tuple length to be 2, but was %s'
        #                      % str(len(_ind)))
    else:
        if isinstance(_ind, np.ndarray):
            if multiline:
                if (_shape is None):
                    raise ValueError(
                        'Shape of array must be given to unravel index.')
                pxpys = np.zeros((_ind.shape[0], 2))
                for i in range(_ind.shape[0]):
                    if _ind[i] == 0:
                        if nozeros:
                            pxpys[i, :] = np.nan, np.nan
                        else:
                            pxpys[i, :] = shared_tools.custom_unravel(_ind[i], _shape)
                    else:
                        pxpys[i, :] = shared_tools.custom_unravel(_ind[i], _shape)
            else:
                if _ind.ndim > 1:
                    pxpys = np.fliplr(_ind)
                else:
                    if (_shape is None):
                        raise ValueError(
                            'Shape of array must be given to unravel index.')
                    pxpys = np.zeros((_ind.shape[0], 2))
                    for i in range(_ind.shape[0]):
                        if _ind[i] == 0:
                            if nozeros:
                                pxpys[i, :] = np.nan, np.nan
                            else:
                                pxpys[i, :] = shared_tools.custom_unravel(_ind[i], _shape)
                        else:
                            pxpys[i, :] = shared_tools.custom_unravel(_ind[i], _shape)

    _l, = ax.plot(pxpys[:, 1], pxpys[:, 0], *args, **kwargs)

    if block:
        plt.show()

    return _l
