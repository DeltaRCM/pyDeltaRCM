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

    def run_one_timestep(self):
        """Run the timestep once.

        The first operation called by :meth:`update`, this method iterates the
        water surface calculation and sediment parcel routing routines.

        .. note:: Will print the current time to stdout, if ``verbose > 0``.

        Parameters
        ----------

        Returns
        -------
        """
        # start the model operations
        #   water iterations
        _msg = 'Beginning water iteration'
        self.log_info(_msg, verbosity=2)
        for iteration in range(self._itermax):

            # initialize the relevant fields and parcel trackers
            self.hook_init_water_iteration()
            self.init_water_iteration()

            # run the actual iteration of the parcels
            self.hook_run_water_iteration()
            self.run_water_iteration()

            # accumulate the routed water parcels into free surface
            self.hook_compute_free_surface()
            self.compute_free_surface()

            # clean up the water surface and apply boundary conditions
            self.hook_finalize_water_iteration(iteration)
            self.finalize_water_iteration(iteration)

        #  sediment iteration
        _msg = 'Beginning sediment iteration'
        self.log_info(_msg, verbosity=2)

        self.hook_sed_route()
        self.sed_route()

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

            self.save_stratigraphy()
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

    def expand_stratigraphy(self):
        """Expand stratigraphy array sizes.

        Parameters
        ----------

        Returns
        -------

        """
        _msg = 'Expanding stratigraphy arrays'
        self.log_info(_msg, verbosity=1)

        lil_blank = lil_matrix((self.L * self.W, self.n_steps),
                               dtype=np.float32)

        self.strata_eta = hstack([self.strata_eta, lil_blank], format='lil')
        self.strata_sand_frac = hstack([self.strata_sand_frac, lil_blank],
                                       format='lil')

    def save_stratigraphy(self):
        """Save stratigraphy to an attribute of the delta.

        Saves the sand fraction of deposited sediment into a sparse array
        created by :obj:`~pyDeltaRCM.DeltaModel.init_stratigraphy()`. Note
        that the actual record in the netcdf file is not made until
        `finalize()`.

        Only runs if :obj:`~pyDeltaRCM.DeltaModel.save_strata` is True.

        .. note::

            This routine needs a complete description of the algorithm,
            additionally, it should be ported to a routine in DeltaMetrics,
            probably the new and preferred method of computing and storing
            straitgraphy cubes.

        Parameters
        ----------

        Returns
        -------

        """
        if self.save_strata:

            if self.strata_counter >= self.strata_eta.shape[1]:
                self.expand_stratigraphy()

            _msg = 'Storing stratigraphy data'
            self.log_info(_msg, verbosity=2)

            # ------------------ sand frac ------------------
            # -1 for cells with deposition volumes < vol_limit
            # vol_limit for any mud (to diff from no deposition in sparse
            #   array)
            # (overwritten if any sand deposited)

            sand_frac = -1 * np.ones((self.L, self.W))

            vol_limit = 0.000001  # threshold deposition volume
            sand_frac[self.Vp_dep_mud > vol_limit] = vol_limit

            sand_loc = self.Vp_dep_sand > 0
            sand_frac[sand_loc] = (self.Vp_dep_sand[sand_loc]
                                   / (self.Vp_dep_mud[sand_loc]
                                      + self.Vp_dep_sand[sand_loc])
                                   )
            # store indices and sand_frac into a sparse array
            row_s = np.where(sand_frac.flatten() >= 0)[0]
            col_s = np.zeros((len(row_s),))
            data_s = sand_frac[sand_frac >= 0]

            sand_sparse = csc_matrix((data_s, (row_s, col_s)),
                                     shape=(self.L * self.W, 1))
            # store sand_sparse into strata_sand_frac
            self.strata_sand_frac[:, self.strata_counter] = sand_sparse

            # ------------------ eta ------------------
            diff_eta = self.eta - self.init_eta

            row_s = np.where(diff_eta.flatten() != 0)[0]
            col_s = np.zeros((len(row_s),))
            data_s = self.eta[diff_eta != 0]

            eta_sparse = csc_matrix((data_s, (row_s, col_s)),
                                    shape=(self.L * self.W, 1))

            self.strata_eta[:, self.strata_counter] = eta_sparse

            if self._toggle_subsidence and (self._time >= self._start_subsidence):

                sigma_change = (self.strata_eta[:, :self.strata_counter]
                                - self.sigma.flatten()[:, np.newaxis])
                self.strata_eta[:, :self.strata_counter] = lil_matrix(
                    sigma_change)

            self.strata_counter += 1

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
        if self._save_any_figs:

            _msg = 'Saving figures'
            self.log_info(_msg, verbosity=2)

            if self._save_eta_figs:
                _fe = self.make_figure('eta', self._time)
                self.save_figure(_fe, directory=self.prefix,
                                 filename_root='eta_',
                                 timestep=self.save_iter)

            if self._save_stage_figs:
                _fs = self.make_figure('stage', self._time)
                self.save_figure(_fs, directory=self.prefix,
                                 filename_root='stage_',
                                 timestep=self.save_iter)

            if self._save_depth_figs:
                _fh = self.make_figure('depth', self._time)
                self.save_figure(_fh, directory=self.prefix,
                                 filename_root='depth_',
                                 timestep=self.save_iter)

            if self._save_discharge_figs:
                _fq = self.make_figure('qw', self._time)
                self.save_figure(_fq, directory=self.prefix,
                                 filename_root='discharge_',
                                 timestep=self.save_iter)

            if self._save_velocity_figs:
                _fu = self.make_figure('uw', self._time)
                self.save_figure(_fu, directory=self.prefix,
                                 filename_root='velocity_',
                                 timestep=self.save_iter)

            if self._save_sedflux_figs:
                _fu = self.make_figure('qs', self._time)
                self.save_figure(_fu, directory=self.prefix,
                                 filename_root='sedflux_',
                                 timestep=self.save_iter)

        # ------------------ grids ------------------
        if self._save_any_grids:

            _msg = 'Saving grids'
            self.log_info(_msg, verbosity=2)

            if self._save_eta_grids:
                self.save_grids('eta', self.eta, save_idx)

            if self._save_depth_grids:
                self.save_grids('depth', self.depth, save_idx)

            if self._save_stage_grids:
                self.save_grids('stage', self.stage, save_idx)

            if self._save_discharge_grids:
                self.save_grids('discharge', self.qw, save_idx)

            if self._save_velocity_grids:
                self.save_grids('velocity', self.uw, save_idx)

            if self._save_sedflux_grids:
                self.save_grids('sedflux', self.qs, save_idx)

            if self._save_discharge_components:
                self.save_grids('discharge_x', self.qx, save_idx)
                self.save_grids('discharge_y', self.qy, save_idx)

            if self._save_velocity_components:
                self.save_grids('velocity_x', self.ux, save_idx)
                self.save_grids('velocity_y', self.uy, save_idx)

        # ------------------ metadata ------------------
        if self._save_metadata:

            _msg = 'Saving metadata'
            self.log_info(_msg, verbosity=2)

            self.output_netcdf['meta']['H_SL'][save_idx] = self._H_SL
            self.output_netcdf['meta']['f_bedload'][save_idx] = self._f_bedload
            self.output_netcdf['meta']['C0_percent'][save_idx] = self._C0_percent
            self.output_netcdf['meta']['u0'][save_idx] = self._u0

        # -------------------- sync --------------------
        if (self._save_metadata or self._save_any_grids):

            _msg = 'Syncing data to output file'
            self.log_info(_msg, verbosity=2)

            self.output_netcdf.sync()

    def record_final_stratigraphy(self):
        """Save the stratigraphy as sparse matrix to file.

        Saves the stratigraphy (sand fraction) sparse matrices into output
        netcdf file.

        .. note:

            This method is called only once, within the
            :obj:`pyDeltaRCM.DeltaModel.finalize()` step of model execution.

        Parameters
        ----------

        Returns
        -------

        """
        if self._save_strata:

            _msg = 'Saving final stratigraphy to netCDF file'
            self.log_info(_msg, verbosity=0)

            if not self.strata_counter > 0:
                _msg = 'Model has no computed stratigraphy. This is likely ' \
                    'because `delta.time < delta.save_dt`, and the model ' \
                    'has not computed stratigraphy.'
                self.logger.error(_msg)
                raise RuntimeError(_msg)

            self.strata_eta = self.strata_eta[:, :self.strata_counter]

            shape = self.strata_eta.shape

            self.output_netcdf.createDimension(
                'total_strata_age',
                shape[1])

            strata_age = self.output_netcdf.createVariable('strata_age',
                                                           np.int32,
                                                           ('total_strata_age'))
            strata_age.units = 'second'
            self.output_netcdf.variables['strata_age'][
                :] = list(range(shape[1] - 1, -1, -1))

            sand_frac = self.output_netcdf.createVariable('strata_sand_frac',
                                                          np.float32,
                                                          ('total_strata_age', 'length', 'width'))
            sand_frac.units = 'fraction'

            strata_elev = self.output_netcdf.createVariable('strata_depth',
                                                            np.float32,
                                                            ('total_strata_age', 'length', 'width'))
            strata_elev.units = 'meters'

            for i in range(shape[1]):

                sf = self.strata_sand_frac[:, i].toarray()
                sf = sf.reshape(self.eta.shape)
                sf[sf == 0] = -1

                self.output_netcdf.variables['strata_sand_frac'][i, :, :] = sf

                sz = self.strata_eta[:, i].toarray().reshape(self.eta.shape)
                sz[sz == 0] = self.init_eta[sz == 0]

                self.output_netcdf.variables['strata_depth'][i, :, :] = sz

            _msg = 'Stratigraphy data saved.'
            self.log_info(_msg, verbosity=0)

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
        ax.set_title(var+'\ntime: '+str(timestep), fontsize=10)

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
        - Current random seed state
        - Stratigraphic 'topography' in 'strata_eta'
        - Stratigraphic sand fraction in 'strata_sand_frac'

        If `save_checkpoint` is turned on, checkpoints are re-written
        with either a frequency of `checkpoint_dt` or `save_dt` if
        `checkpoint_dt` has not been explicitly defined.
        """
        ckp_file = os.path.join(self.prefix, 'checkpoint.npz')

        # handle checkpoint which needs stratigraphy info
        strata_dict = {}
        if self.save_strata:
            # convert sparse arrays to csr type so they are easier to save
            csr_strata_eta = self.strata_eta.tocsr()
            csr_strata_sand_frac = self.strata_sand_frac.tocsr()

            strata_dict.update(
                {'eta_data': csr_strata_eta.data,
                 'eta_indices': csr_strata_eta.indices,
                 'eta_indptr': csr_strata_eta.indptr,
                 'eta_shape': csr_strata_eta.shape,
                 'sand_data': csr_strata_sand_frac.data,
                 'sand_indices': csr_strata_sand_frac.indices,
                 'sand_indptr': csr_strata_sand_frac.indptr,
                 'sand_shape': csr_strata_sand_frac.shape,
                 'n_steps': self.n_steps,
                 'init_eta': self.init_eta,
                 'strata_counter': self.strata_counter
                 }
                )

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
            # boundary condition / state variables
            H_SL=self._H_SL,
            rng_state=rng_state,
            # stratigraphy related variables
            **strata_dict
            )
