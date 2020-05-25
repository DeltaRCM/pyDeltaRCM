#! /usr/bin/env python
import warnings
import logging
from .deltaRCM_tools import Tools
import datetime
import os


class pyDeltaRCM(Tools):

    def __init__(self, input_file=None):
        """Creates an instance of the pyDeltaRCM model.

        This method handles setting up the run, including parsing input files,
        initializing arrays, and initializing output routines.

        Parameters
        ----------
        input_file : `str`, `os.PathLike`, optional
            User model run configuration file.

        Notes
        -----
        For more information regarding input configuration files, see the
        :doc:`../../guides/userguide`.

        """

        self._time = 0.
        self._time_step = 1.

        self.input_file = input_file
        self._file_dir = os.path.realpath(os.path.dirname(__file__))
        self.default_file = os.path.join(self._file_dir, 'default.yml')

        self.import_files()

        self.create_other_variables()

        self.init_logger()

        self.create_domain()

        self.init_subsidence()
        self.init_stratigraphy()
        self.init_output_grids()

    def update(self):
        """
        Run the model for one full instance
        """
        self.run_one_timestep()

        self.apply_subsidence()

        self.finalize_timestep()
        self.record_stratigraphy()

        self.output_data()

        self._time += self.time_step

    def finalize(self):

        self.output_strata()

        try:
            self.output_netcdf.close()
            if self.verbose >= 1:
                print('Closed output netcdf file.')
        except Exception:
            pass

        self._is_finalized = True

    # define properties

    @property
    def time_step(self):
        """The time step."""
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
        """ Get channel flow velocity """
        return self.u0

    @channel_flow_velocity.setter
    def channel_flow_velocity(self, new_u0):
        self.u0 = new_u0
        self.create_other_variables()

    @property
    def channel_width(self):
        """ Get channel width """
        return self.N0_meters

    @channel_width.setter
    def channel_width(self, new_N0):
        self.N0_meters = new_N0
        self.create_other_variables()

    @property
    def channel_flow_depth(self):
        """ Get channel flow depth """
        return self.h0

    @channel_flow_depth.setter
    def channel_flow_depth(self, new_d):
        self.h0 = new_d
        self.create_other_variables()

    @property
    def sea_surface_mean_elevation(self):
        """ Get sea surface mean elevation """
        return self.H_SL

    @sea_surface_mean_elevation.setter
    def sea_surface_mean_elevation(self, new_se):
        self.H_SL = new_se

    @property
    def sea_surface_elevation_change(self):
        """ Get rate of change of sea surface elevation, per timestep"""
        return self.SLR

    @sea_surface_elevation_change.setter
    def sea_surface_elevation_change(self, new_SLR):
        """ Set rate of change of sea surface elevation, per timestep"""
        self.SLR = new_SLR

    @property
    def bedload_fraction(self):
        """ Get bedload fraction """
        return self.f_bedload

    @bedload_fraction.setter
    def bedload_fraction(self, new_u0):
        self.f_bedload = new_u0

    @property
    def influx_sediment_concentration(self):
        """ Get influx sediment concentration """
        return self.C0_percent

    @influx_sediment_concentration.setter
    def influx_sediment_concentration(self, new_u0):
        self.C0_percent = new_u0
        self.create_other_variables()

    @property
    def sea_surface_elevation(self):
        """ Get stage """
        return self.stage

    @property
    def water_depth(self):
        """ Get depth """
        return self.depth

    @property
    def bed_elevation(self):
        """ Get bed elevation """
        return self.eta
