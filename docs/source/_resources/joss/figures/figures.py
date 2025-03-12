import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import sys

import pyDeltaRCM
from pyDeltaRCM.shared_tools import sec_in_day, day_in_yr

import netCDF4 as nc
from typing import List

# everything relative to this file
_dir: str = os.path.realpath(os.path.dirname(__file__))


if __name__ == '__main__':

    # get script input argument
    _arg: List[str] = sys.argv
    if len(_arg) == 3:
        _input_flag: str = sys.argv[1].strip('-')
    elif len(_arg) == 2:
        _input_flag: str = sys.argv[1].strip('-')
    else:
        raise ValueError('No arguments supplied.')

    # parameter choices for scaling
    If: float = 10 / day_in_yr  # intermittency factor for year-scaling

    # if running the computation
    if _input_flag == 'compute':

        _mdl = pyDeltaRCM.DeltaModel(
            out_dir=_dir,
            seed=10151919,  # JOSS, A1Z26 encoded
            save_eta_grids=True,
            save_dt=sec_in_day)

        for i in range(4000):
            _mdl.update()

        # finalize
        _mdl.finalize()

    # if running the plotting
    elif _input_flag == 'plot':

        # set up attributes needed for plotting
        H_SL, h, n = 0, 4, 0.3  # sea level, basin depth, surf relief
        blues = matplotlib.cm.get_cmap('Blues_r', 64)
        greens = matplotlib.cm.get_cmap('YlGn_r', 64)
        combined = np.vstack((blues(np.linspace(0.1, 0.7, 5)),
                              greens(np.linspace(0.2, 0.8, 5))))
        cmap = matplotlib.colors.ListedColormap(combined, name='delta')
        bounds = np.hstack(
            (np.linspace(H_SL-h, H_SL-(n/2), 5),
             np.linspace(H_SL, H_SL+n, 6)))
        norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds)-1)

        data = nc.Dataset(os.path.join(_dir, 'pyDeltaRCM_output.nc'))

        nt = 4
        ts = np.linspace(0, data['eta'].shape[0]-1,
                         num=nt, dtype=int)  # linearly interpolate ts

        # make the timeseries plot
        fig, ax = plt.subplots(1, nt, figsize=(11, 2), dpi=300)
        for i, t in enumerate(ts):
            ax[i].imshow(data['eta'][t, :, :], cmap=cmap, norm=norm)
            _time = pyDeltaRCM.shared_tools.scale_model_time(
                data['time'][t], If=If, units='years')
            ax[i].set_title(' '.join((
                str(np.round(_time, 1)),
                'years')))
            ax[i].axes.get_xaxis().set_ticks([])
            ax[i].axes.get_yaxis().set_ticks([])

        # plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(_dir, 'timeseries.png'), transparent=True)

    # otherwise
    else:
        raise ValueError('Invalid argument supplied.')
