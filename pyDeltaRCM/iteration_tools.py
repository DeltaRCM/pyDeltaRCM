#! /usr/bin/env python

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axtk

from scipy.sparse import lil_matrix, csc_matrix, hstack

import abc

from . import shared_tools


class iteration_tools(abc.ABC):
    """Tools relating to the updating of the model and model I/O.

    Tools defined in this class include steps to iterate for one timestep,
    finalize timesteps, and saving output figures, grids, and checkpoints.
    Additionally, most stratigraphy-related operations are defined here, since
    these operations largely occur when saving and updating the model.
    """

    def solve_water_and_sediment_timestep(self):
        """Run water and sediment operations for one timestep.

        The first operation called by :meth:`update`, this method iterates the
        water surface calculation and sediment parcel routing routines.

        Parameters
        ----------

        Returns
        -------
        """
        # start the model operations
        self.eta0 = np.copy(self.eta)  # copy

        # water iterations
        self.hook_route_water()
        self.route_water()

        # sediment iteration
        self.hook_route_sediment()
        self.route_sediment()

    def run_one_timestep(self):
        """Deprecated, since v1.3.1. Use :obj:`solve_water_and_sediment_timestep`."""
        _msg = ('`run_one_timestep` and `hook_run_one_timestep` are '
                'deprecated and have been replaced with '
                '`solve_water_and_sediment_timestep`. '
                'Running `solve_water_and_sediment_timestep` now, but '
                'this will be removed in future release.')
        self.logger.warning(_msg)
        warnings.warn(UserWarning(_msg))
        self.solve_water_and_sediment_timestep()

    def apply_subsidence(self):
        """Apply subsidence pattern.

        Apply subsidence to domain if toggle_subsidence is True, and
        :obj:`~pyDeltaRCM.DeltaModel.time` is ``>=``
        :obj:`~pyDeltaRCM.DeltaModel.start_subsidence`. Note, that the
        configuration of the :obj:`~pyDeltaRCM.DeltaModel.update()` method
        determines that the subsidence may be applied before the model time
        is incremented, such that subsidence will begin on the step
        *following* the time step that brings the model to ``time ==
        start_subsidence``.

        Parameters
        ----------

        Returns
        -------

        """
        if self._toggle_subsidence:

            if self._time >= self._start_subsidence:

                _msg = 'Applying subsidence'
                self.log_info(_msg, verbosity=1)

                self.eta[:] = self.eta - self.sigma

    def finalize_timestep(self):
        """Finalize timestep.

        Clean up after sediment routing. This includes a correction for
        flooded cells that are not "wet" (via :meth:`flooding_correction`).

        Update sea level if baselevel changes between timesteps.

        Parameters
        ----------

        Returns
        -------

        """
        _msg = 'Finalizing timestep'
        self.log_info(_msg, verbosity=2)

        self.flooding_correction()
        self.stage[:] = np.maximum(self.stage, self._H_SL)
        self.depth[:] = np.maximum(self.stage - self.eta, 0)

        self.eta[0, self.inlet] = self.stage[0, self.inlet] - self._h0
        self.depth[0, self.inlet] = self._h0

        self.hook_compute_sand_frac()
        self.compute_sand_frac()

        self.H_SL = self._H_SL + self._SLR * self._dt

    def log_info(self, message, verbosity=0):
        """Log message dependent on verbosity settings.

        Parameters
        ----------
        message : :obj:`str`
            Message string to write to the log as info.

        verbosity : :obj:`int`, optional
            Verbosity threshold, whether to write the message to the log or
            not. Default value is `0`, or i.e. to always log.
        """
        if self._verbose >= verbosity:
            self.logger.info(message)

    def log_model_time(self):
        """Log the time of the model.

        Reports the time to the log file, and depending on verbosity, will
        report it to stdout.
        """
        _timemsg = 'Time: {time:.{digits}f}; timestep: {timestep:g}'.format(
            time=self._time, timestep=self._time_iter, digits=1)
        self.logger.info(_timemsg)
        if self._verbose > 0:
            print(_timemsg)

    def output_data(self):
        """Output grids and figures if needed.

        """
        if self._save_time_since_data >= self.save_dt:

            self.save_grids_and_figs()

            self._save_iter += int(1)
            self._save_time_since_data = 0

    def output_checkpoint(self):
        """Output checkpoint if needed.

        Save checkpoint data (including rng state) so that the model can be
        resumed from this time.

        Parameters
        ----------

        Returns
        -------

        """
        if self._save_time_since_checkpoint >= self.checkpoint_dt:

            if self._save_checkpoint:

                _msg = 'Saving checkpoint'
                self.log_info(_msg, verbosity=1)

                self.save_the_checkpoint()

                if self._checkpoint_dt != self._save_dt:
                    _msg = ('Grid save interval and checkpoint interval are '
                            'not identical, this may result in duplicate '
                            'entries in the output NetCDF4 after resuming '
                            'the model run.')
                    self.logger.warning(_msg)

                self._save_time_since_checkpoint = 0

    def compute_sand_frac(self):
        """Compute the sand fraction as a continous updating data field.

        Parameters
        ----------

        Returns
        -------

        """
        _msg = 'Computing bed sand fraction'
        self.log_info(_msg, verbosity=2)

        # layer attributes at time t
        actlyr_thick = self._active_layer_thickness
        actlyr_top = np.copy(self.eta0)
        actlyr_bot = actlyr_top - actlyr_thick

        deta = self.eta - self.eta0

        # everywhere the bed has degraded this timestep
        whr_deg = (deta < 0)
        if np.any(whr_deg):
            # find where the erosion exceeded the active layer
            whr_unkwn = self.eta < actlyr_bot

            # update sand_frac in unknown to the boundary condition
            self.sand_frac[whr_unkwn] = self._sand_frac_bc

            # find where erosion was into active layer
            whr_actero = np.logical_and(whr_deg, self.eta >= actlyr_bot)

            # update sand_frac to active_layer value
            self.sand_frac[whr_actero] = self.active_layer[whr_actero]

        # handle aggradation/deposition
        whr_agg = (deta > 0)
        whr_agg = np.logical_or(
            (self.Vp_dep_sand > 0), (self.Vp_dep_mud > 0.000001))
        if np.any(whr_agg):
            # sand_frac and active_layer becomes the mixture of the deposit
            mixture = (self.Vp_dep_sand[whr_agg] /
                       (self.Vp_dep_mud[whr_agg] +
                        self.Vp_dep_sand[whr_agg]))

            # update sand_frac in act layer to this value
            self.sand_frac[whr_agg] = mixture
            self.active_layer[whr_agg] = mixture

    def save_grids_and_figs(self):
        """Save grids and figures.

        Save grids and/or plots of specified variables (``eta``, `discharge``,
        ``velocity``, ``depth``, and ``stage``, depending on configuration of
        the relevant flags in the YAML configuration file.

        .. note:

            This method is called often throughout the model, each
            occurance :obj:`save_dt` is elapsed in model time.

        Parameters
        ----------

        Returns
        -------

        """
        save_idx = self.save_iter

        _msg = ' '.join((
            'Saving data to output file:',
            str(save_idx).zfill(5)))
        self.log_info(_msg, verbosity=1)

        if (self._save_metadata or self._save_any_grids):
            self.output_netcdf.variables['time'][save_idx] = self._time

        # ------------------ Figures ------------------
        if len(self._save_fig_list) > 0:

            _msg = 'Saving figures'
            self.log_info(_msg, verbosity=2)

            for f in self._save_fig_list.keys():
                _attr = getattr(self, self._save_fig_list[f][0])
                if isinstance(_attr, np.ndarray):
                    if _attr.shape == (self.L, self.W):
                        _fig = self.make_figure(self._save_fig_list[f][0],
                                                self._time)
                        self.save_figure(_fig, directory=self.prefix,
                                         filename_root=f+'_',
                                         timestep=self.save_iter)
                    else:
                        raise AttributeError('Attribute "{_k}" is not of the '
                                             'right shape to be saved as a '
                                             'figure using the built-in '
                                             'methods. Expected a shape of '
                                             '"{_expshp}", but it has a shape '
                                             'of "{_wasshp}". Consider making '
                                             'a custom plotting utility to '
                                             'visualize this attribute.'
                                             .format(_k=f,
                                                     _expshp=(self.L, self.W),
                                                     _wasshp=_attr.shape))
                else:
                    raise AttributeError('Only plotting of np.ndarray-type '
                                         'attributes is natively supported. '
                                         'Input "{_k}" was of type "{_wt}".'
                                         .format(_k=f, _wt=type(_attr)))

        # ------------------ grids ------------------
        if self._save_any_grids:

            _msg = 'Saving grids'
            self.log_info(_msg, verbosity=2)

            _var_list = list(self._save_var_list.keys())
            _var_list.remove('meta')
            for _val in _var_list:
                self.save_grids(_val, getattr(self,
                                              self._save_var_list[_val][0]),
                                save_idx)

        # ------------------ metadata ------------------
        if self._save_metadata:

            _msg = 'Saving metadata'
            self.log_info(_msg, verbosity=2)

            for _val in self._save_var_list['meta'].keys():
                # use knowledge of time-varying values to save them
                if (self._save_var_list['meta'][_val][0] is None):
                    self.output_netcdf['meta'][_val][save_idx] = \
                        getattr(self, _val)

        # -------------------- sync --------------------
        if (self._save_metadata or self._save_any_grids):

            _msg = 'Syncing data to output file'
            self.log_info(_msg, verbosity=2)

            self.output_netcdf.sync()

    def make_figure(self, var, timestep):
        """Create a figure.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to plot into the figure. Specified as a string and
            looked up via `getattr`.

        Returns
        -------
        fig : :obj:`matplotlib.figure`
            The created figure object.
        """
        _data = getattr(self, var)

        fig, ax = plt.subplots()
        im = ax.pcolormesh(self.X, self.Y, _data, shading='flat')
        ax.set_xlim((0, self._Width))
        ax.set_ylim((0, self._Length))
        ax.set_aspect('equal', adjustable='box')
        divider = axtk.axes_divider.make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7)
        ax.use_sticky_edges = False
        ax.margins(y=0.2)
        ax.set_title(str(var)+'\ntime: '+str(timestep), fontsize=10)

        return fig

    def save_figure(self, fig, directory, filename_root,
                    timestep, ext='.png', close=True):
        """Save a figure.

        Parameters
        ----------
        path : :obj:`str`
            The path (and filename without extension) to save the figure to.

        ext : :obj:`str`, optional
            The file extension (default='.png'). This must be supported by the
            active matplotlib backend (see matplotlib.backends module). Most
            backends support '.png', '.pdf', '.ps', '.eps', and '.svg'. Be
            sure to include the '.' before the extension.

        close : :obj:`bool`, optional
            Whether to close the file after saving.

        Returns
        -------

        """
        if self._save_figs_sequential:
            # save as a padded number with the timestep
            savepath = os.path.join(directory,
                                    filename_root + str(timestep).zfill(5) + ext)
        else:
            # save as "latest"
            savepath = os.path.join(directory,
                                    filename_root + 'latest' + ext)

        fig.savefig(savepath)
        if close:
            plt.close()

    def save_grids(self, var_name, var, ts):
        """Save a grid into an existing netCDF file.

        File should already be open (by :meth:`init_output_grid`) as
        ``self.output_netcdf``.

        Parameters
        ----------
        var_name : :obj:`str`
            The name of the variable to be saved

        var : :obj:`ndarray`
            The numpy array to be saved.

        ts : :obj:`int`
            The current timestep (+1, so human readable)

        Returns
        -------

        """
        _msg = ' '.join(['saving', str(var_name), 'grid'])
        self.log_info(_msg, verbosity=2)
        try:
            self.output_netcdf.variables[var_name][ts, :, :] = var
        except:
            _msg = 'Failed to save {var_name} grid to netCDF file.'.format(var_name=var_name)
            self.logger.error(_msg)
            warnings.warn(UserWarning('Cannot save grid to netCDF file.'))

    def save_the_checkpoint(self):
        """Save checkpoint files.

        Saves the grids to a .npz file so that the model can be
        initiated from this point. The timestep of the checkpoint is also
        saved. The values from the model that are saved to the checkpoint.npz
        are the following:

        - Model time
        - Flow velocity and its components
        - Water depth
        - Water stage
        - Topography
        - Surface sand fraction
        - Active layer values
        - Current random seed state

        If `save_checkpoint` is turned on, checkpoints are re-written
        with either a frequency of `checkpoint_dt` or `save_dt` if
        `checkpoint_dt` has not been explicitly defined.
        """
        ckp_file = os.path.join(self.prefix, 'checkpoint.npz')

        # get rng state
        rng_state_list = shared_tools.get_random_state()
        rng_state = np.array(rng_state_list,
                             dtype=object)  # convert to object before saving

        np.savez_compressed(
            ckp_file,
            # time and counter variables
            time=self.time,
            time_iter=self._time_iter,
            save_iter=self._save_iter,
            save_time_since_data=self._save_time_since_data,
            # grids
            eta=self.eta,
            depth=self.depth,
            stage=self.stage,
            uw=self.uw,
            ux=self.ux,
            uy=self.uy,
            qw=self.qw,
            qx=self.qx,
            qy=self.qy,
            sand_frac=self.sand_frac,
            active_layer=self.active_layer,
            # boundary condition / state variables
            H_SL=self._H_SL,
            rng_state=rng_state,
            )
