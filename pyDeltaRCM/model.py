#! /usr/bin/env python
import warnings
import logging
from .deltaRCM_tools import Tools
import datetime
import os


class DeltaModel(Tools):
    """Main model class.

    Instantiating the model class is described in the :meth:`__init__` method
    below in detail, but generally model instantiation occurs via a model run
    YAML configuration file. These YAML configuration files define model
    parameters which are used in the run; read more about creating input YAML
    configuration files in the :doc:`../../guides/userguide`.

    Once the model has been instantiated, the model is updated via the
    :meth:`update`. This method coordinates the hydrology, sediment transport,
    subsidence, and stratigraphic operation that must occur during each
    timestep. The :meth:`update` should be called iteratively, to "run" the
    model.

    Finally, after all ``update`` steps are complete, the model should be
    finalized (:meth:`finalize`), such that files and data are appropriately
    closed and written to disk.
    """

    def __init__(self, input_file=None):
        """Creates an instance of the pyDeltaRCM model.

        This method handles setting up the run, including parsing input files,
        initializing arrays, and initializing output routines.

        Parameters
        ----------
        input_file : `str`, `os.PathLike`, optional
            User model run configuration file.

        Returns
        -------

        Notes
        -----
        For more information regarding input configuration files, see the
        :doc:`../../guides/userguide`.

        """
        self._time = 0.
        self._time_step = 1.

        self.input_file = input_file
        _src_dir = os.path.realpath(os.path.dirname(__file__))
        self.default_file = os.path.join(_src_dir, 'default.yml')
        self.import_files()

        self.init_output_infrastructure()
        self.init_logger()

        self.create_other_variables()

        self.determine_random_seed()
        self.create_domain()

        self.init_subsidence()
        self.init_stratigraphy()
        self.init_output_grids()

        self.logger.info('Model initialization complete')

    def update(self):
        """Run the model for one full instance

        Calls, in sequence:

            * the routine to run one timestep (i.e., water surface estimation
              and sediment routing, :meth:`run_one_timestep`)
            * the basin subsidence update pattern (:meth:`apply_subsidence`)
            * the timestep finalization routine (:meth:`finalize_timestep`)
            * straigraphy updating routine (:meth:`record_stratigraphy`)

        If you attempt to override the ``update`` routine, you must implement
        these operations at a minimum.

        Parameters
        ----------

        Returns
        -------

        """
        self.run_one_timestep()

        self.apply_subsidence()

        self.finalize_timestep()
        self.record_stratigraphy()

        self.output_data()

        self._time += self.time_step

    def finalize(self):
        """Finalize the model run.

        Finalization includes saving output stratigraphy, closing relevant
        files on disk and clearing variables.

        Parameters
        ----------

        Returns
        -------

        """
        self.logger.info('Finalize model run')
        self.output_strata()

        try:
            self.output_netcdf.close()
            _msg = 'Closed output netcdf file'
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)
        except Exception:
            pass

        self._is_finalized = True

    @property
    def time_step(self):
        """The time step.

        Raises
        ------
        UserWarning
            If a very small timestep is configured.
        """
        return self._time_step

    @time_step.setter
    def time_step(self, new_dt):
        if new_dt * self.init_Np_sed < 100:
            warnings.warn(UserWarning('Using a very small timestep, '
                                      'Delta might evolve very slowly.'))

        self.Np_sed = int(new_dt * self.init_Np_sed)
        self.Np_water = int(new_dt * self.init_Np_water)

        if self.toggle_subsidence:
            self.sigma = self.subsidence_mask * self.sigma_max * new_dt

        self._time_step = new_dt

    @property
    def channel_flow_velocity(self):
        """Get channel flow velocity."""
        return self.u0

    @channel_flow_velocity.setter
    def channel_flow_velocity(self, new_u0):
        self.u0 = new_u0
        self.create_other_variables()

    @property
    def channel_width(self):
        """Get channel width."""
        return self.N0_meters

    @channel_width.setter
    def channel_width(self, new_N0):
        self.N0_meters = new_N0
        self.create_other_variables()

    @property
    def channel_flow_depth(self):
        """Get channel flow depth."""
        return self.h0

    @channel_flow_depth.setter
    def channel_flow_depth(self, new_d):
        self.h0 = new_d
        self.create_other_variables()

    @property
    def sea_surface_mean_elevation(self):
        """Get sea surface mean elevation."""
        return self.H_SL

    @sea_surface_mean_elevation.setter
    def sea_surface_mean_elevation(self, new_se):
        self.H_SL = new_se

    @property
    def sea_surface_elevation_change(self):
        """Get rate of change of sea surface elevation, per timestep."""
        return self.SLR

    @sea_surface_elevation_change.setter
    def sea_surface_elevation_change(self, new_SLR):
        self.SLR = new_SLR

    @property
    def bedload_fraction(self):
        """Get bedload fraction."""
        return self.f_bedload

    @bedload_fraction.setter
    def bedload_fraction(self, new_u0):
        self.f_bedload = new_u0

    @property
    def influx_sediment_concentration(self):
        """Get influx sediment concentration."""
        return self.C0_percent

    @influx_sediment_concentration.setter
    def influx_sediment_concentration(self, new_u0):
        self.C0_percent = new_u0
        self.create_other_variables()

    @property
    def sea_surface_elevation(self):
        """Get stage."""
        return self.stage

    @property
    def water_depth(self):
        """Get depth."""
        return self.depth

    @property
    def bed_elevation(self):
        """Get bed elevation."""
        return self.eta
