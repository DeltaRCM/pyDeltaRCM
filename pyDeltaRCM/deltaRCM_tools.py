#! /usr/bin/env python

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csc_matrix, hstack

from .sed_tools import sed_tools
from .water_tools import water_tools
from .init_tools import init_tools
from .debug_tools import debug_tools


class Tools(sed_tools, water_tools, init_tools, debug_tools, object):
    """Aggregate all `*_tools` modules into single class, define api methods."""

    def run_one_timestep(self):
        """Run the timestep once.

        The first operation called by :meth:`update`, this method iterates the
        water surface calculation and sediment parcel routing routines.

        .. note:: Will print the current timestep to stdout, if ``verbose > 0``.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        RuntimeError 
            If model has already been finalized via :meth:`finalize`.
        """
        timestep = self._time

        self.logger.info('-' * 4 + ' Timestep ' +
                         str(self._time) + ' ' + '-' * 4)
        if self.verbose > 0:
            print('-' * 20)
            print('Timestep: ' + str(self._time))

        if self._is_finalized:
            raise RuntimeError('Cannot update model, model already finalized!')

        # model operations
        for iteration in range(self.itermax):

            self.init_water_iteration()
            self.run_water_iteration()

            self.free_surf(iteration)
            self.finalize_water_iteration(timestep, iteration)

        self.sed_route()

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
        self.flooding_correction()
        self.stage[:] = np.maximum(self.stage, self.H_SL)
        self.depth[:] = np.maximum(self.stage - self.eta, 0)

        self.eta[0, self.inlet] = self.stage[0, self.inlet] - self.h0
        self.depth[0, self.inlet] = self.h0

        self.H_SL = self.H_SL + self.SLR * self.dt

    def expand_stratigraphy(self):
        """Expand stratigraphy array sizes.

        Parameters
        ----------

        Returns
        -------

        """
        _msg = 'Expanding stratigraphy arrays'
        self.logger.info(_msg)
        if self.verbose >= 2:
            print(_msg)

        lil_blank = lil_matrix((self.L * self.W, self.n_steps),
                               dtype=np.float32)

        self.strata_eta = hstack([self.strata_eta, lil_blank], format='lil')
        self.strata_sand_frac = hstack([self.strata_sand_frac, lil_blank],
                                       format='lil')

    def record_stratigraphy(self):
        """Save stratigraphy to file.

        Saves the sand fraction of deposited sediment
        into a sparse array created by init_stratigraphy().

        Only runs if save_strata is True.

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
        timestep = self._time

        if self.save_strata and (timestep % self.save_dt == 0):

            timestep = int(timestep)

            if self.strata_eta.shape[1] <= timestep:
                self.expand_stratigraphy()

            _msg = 'Storing stratigraphy data'
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)

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

            if self.toggle_subsidence and self.start_subsidence <= timestep:

                sigma_change = (self.strata_eta[:, :self.strata_counter]
                                - self.sigma.flatten()[:, np.newaxis])
                self.strata_eta[:, :self.strata_counter] = lil_matrix(
                    sigma_change)

            self.strata_counter += 1

    def apply_subsidence(self):
        """Apply subsidence pattern.

        Apply subsidence to domain if toggle_subsidence is True, and
        start_subsidence is =< timestep.

        Parameters
        ----------

        Returns
        -------

        """
        if self.toggle_subsidence:

            timestep = self._time

            if self.start_subsidence <= timestep:

                _msg = 'Applying subsidence'
                self.logger.info(_msg)
                if self.verbose >= 2:
                    print(_msg)

                self.eta[:] = self.eta - self.sigma

    def output_data(self):
        """Save grids and figures.

        Save grids and/or plots of specified variables (``eta``, `discharge``,
        ``velocity``, ``depth``, and ``stage``, depending on configuration of
        the relevant flags in the YAML configuration file.

        Parameters
        ----------

        Returns
        -------

        """
        _timestep = self._time
        if _timestep % self.save_dt == 0:

            _timestep = int(self._time)

            # ------------------ Figures ------------------
            if self._save_any_figs:

                _msg = 'Saving figures'
                self.logger.info(_msg)
                if self.verbose >= 2:
                    print(_msg)

                if self.save_eta_figs:
                    _fe = self.make_figure('eta', _timestep)
                    self.save_figure(_fe, directory=self.prefix,
                                     filename_root='eta_',
                                     timestep=_timestep)

                if self.save_stage_figs:
                    _fs = self.make_figure('stage', _timestep)
                    self.save_figure(_fs, directory=self.prefix,
                                     filename_root='stage_',
                                     timestep=_timestep)

                if self.save_depth_figs:
                    _fh = self.make_figure('depth', _timestep)
                    self.save_figure(_fh, directory=self.prefix,
                                     filename_root='depth_',
                                     timestep=_timestep)

                if self.save_discharge_figs:
                    _fq = self.make_figure('qw', _timestep)
                    self.save_figure(_fq, directory=self.prefix,
                                     filename_root='discharge_',
                                     timestep=_timestep)

                if self.save_velocity_figs:
                    _fu = self.make_figure('uw', _timestep)
                    self.save_figure(_fu, directory=self.prefix,
                                     filename_root='velocity_',
                                     timestep=_timestep)

            # ------------------ grids ------------------
            if self._save_any_grids:

                shape = self.output_netcdf.variables['time'].shape
                self.output_netcdf.variables['time'][shape[0]] = _timestep

                _msg = 'Saving grids'
                self.logger.info(_msg)
                if self.verbose >= 2:
                    print(_msg)

                if self.save_eta_grids:
                    self.save_grids('eta', self.eta, shape[0])

                if self.save_depth_grids:
                    self.save_grids('depth', self.depth, shape[0])

                if self.save_stage_grids:
                    self.save_grids('stage', self.stage, shape[0])

                if self.save_discharge_grids:
                    self.save_grids('discharge', self.qw, shape[0])

                if self.save_velocity_grids:
                    self.save_grids('velocity', self.uw, shape[0])

    def output_strata(self):
        """Save stratigraphy as sparse matrix to file.

        Saves the stratigraphy (sand fraction) sparse matrices into output netcdf file

        Parameters
        ----------

        Returns
        -------

        """
        if self.save_strata:

            _msg = 'Saving final stratigraphy to netCDF file'
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)

            self.strata_eta = self.strata_eta[:, :self.strata_counter]

            shape = self.strata_eta.shape
            if shape[0] < 1:
                raise RuntimeError('Stratigraphy are empty! '
                                   'Are you sure you ran the model at least '
                                   'one timestep with `update()`?')

            total_strata_age = self.output_netcdf.createDimension(
                'total_strata_age',
                shape[1])

            strata_age = self.output_netcdf.createVariable('strata_age',
                                                           np.int32,
                                                           ('total_strata_age'))
            strata_age.units = 'timesteps'
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
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)

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
        pc = ax.pcolor(_data)
        fig.colorbar(pc)
        ax.set_title(var + ' --- ' + 'time = ' + str(timestep))
        ax.axis('equal')

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
        try:
            self.output_netcdf.variables[var_name][ts, :, :] = var
        except:
            self.logger.error('Cannot save grid to netCDF file.')
            warnings.warn(UserWarning('Cannot save grid to netCDF file.'))
