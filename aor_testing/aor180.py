import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import shutil
import sys
import yaml

import scipy as sp

import pyDeltaRCM
from pyDeltaRCM.shared_tools import sec_in_day, day_in_yr, custom_pad

class aor180Model(pyDeltaRCM.DeltaModel):
    def __init__(self, input_file, **kwargs):
        # inherit from base model
        super().__init__(input_file, **kwargs)
        # hook_after_create_domain is called here
        pass

    def hook_import_files(self):
        """Define the custom YAML parameters."""
        # custom numeric parameter
        self.subclass_parameters['fail_deg_crit'] = {
            'type': ['float'], 'default': 30.
        }
       
        self.subclass_parameters['dep_deg_crit'] = {
            'type': ['float'], 'default': 14.
        }
        self.total_failure_volume = 0.0

        pass

    def init_output_file(self):
        super().init_output_file()
        """Add non-standard grids, figures and metadata to be saved."""
        if self._save_metadata or self._save_any_grids:

            if 'total_failure_volume' not in self.output_netcdf.variables:
                var = self.output_netcdf.createVariable('total_failure_volume', 'f8', ('time',))
                var.units = 'm^3/t'

       
        # save the slopes at which deposition and failure occur in degrees as metadata
        self._save_var_list['meta']['dep_deg_crit'] = ['dep_deg_crit',
                                                        'degrees',
                                                        'f8', ()]
        self._save_var_list['meta']['fail_deg_crit'] = ['fail_deg_crit',
                                                        'degrees',
                                                        'f8', ()]
        pass

    def hook_output_data(self):
        if self._save_metadata or self._save_any_grids:
            # `self.save_iter` tracks the current NetCDF time index (0, 1, 2...)
            self.output_netcdf.variables['total_failure_volume'][self.save_iter] = self.total_failure_volume
        pass

    def hook_topo_diffusion(self, **kwargs):
       # if self._time_iter % 1 == 0: 
        self.angle_of_repose()
        pass

    def angle_of_repose(self):
        """Routine for foreset processes with given dimensions."""
        """
        Code block 1, here we calculate the local slope maxima and their indices
        """
        distances = self.distances_flat # distances between each cell
        pad_eta = custom_pad(self.eta) #pad cells with nearby neighbors
        slope = np.zeros((self.L, self.W, 9)) #Construct an array with basin dimesions filled with vectors of 9 zeros
        #Calculate slope of each cell and surrounding neighbors and fill into vectors at each cells
        for i in np.arange(self.L):
            for j in np.arange(self.W):
                eta_nbrs = pad_eta[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]
                slope[i, j, :] = np.tan(
                    (self.eta[i, j] - eta_nbrs.ravel()) / (distances * self.dx)
                )  # units are radians
        dirmax = np.argmax(slope, axis=2) #find local slope maxima in each set of neighbor cells
        m, n = dirmax.shape #find coordinates of local slope maxima in each set of neighbor cells
        I, J = np.ogrid[:m, :n] #make boolean grid of which neighbor cell is local slope maxima at cell
        slopemax = slope[I, J, dirmax] #

        """
        Code block 2
        Implement a router, in this case based on angle of repose
        basically:
          find places where the slope is steeper than some critical value
          randomly select them in some order
          calculate the thickness at that cell that is unstable (i.e., how much removed would lower the slope to below threshold)
          move that volume of sediment down the slope in the max direction
          in the future, could implement complex routing rules to approx diff turbidity currents?
        """        

        # critical slope controls the size of the failure and therefore the
        # coherence of the mass moving down the slope, and how far (indirectly) the mass moves before depositing.
        fail_deg_crit = self.fail_deg_crit  # critical slope for failure in degrees
        dep_deg_crit = self.dep_deg_crit  # critical slope for deposition in degrees
        # eventually have those as yaml parameters if we want have degrees set up already?
        fail_slope_crit = np.radians(fail_deg_crit)  # units as radians
        dep_slope_crit = np.radians(dep_deg_crit)

        steep = slopemax > fail_slope_crit
        cell = self.cell_type == 0
        whr_steep = np.where(np.logical_and(steep, cell))
        # reset total failure volume from proceeding timestep
        self.total_failure_volume = 0
        # compute failure where it's steep
        for i, (ix, iy) in enumerate(zip(*whr_steep)):
            pad_eta = custom_pad(self.eta)  # recompute on each transport

            idir = dirmax[ix, iy]
            #calculate failure thickness for an individual cell based on the volume in that cell above the deposition angle
            #maybe come back and change this later?
            #
            current_excess_height = np.tan(fail_slope_crit) * self.dx
            new_excess_height = np.tan(dep_slope_crit) * self.dx
            failure_thickness = current_excess_height - new_excess_height
            # TODO: limit failure thickness to initial bedrock
            # deposit_thickness = (
            #     self.eta[ix, iy] - self.eta0[ix, iy]
            # )  # could vectorize above loop

            #calculate the failure volume 
            failure_volume = failure_thickness * self.dx * self.dx
            self.total_failure_volume += failure_volume
            self.eta[ix, iy] = self.eta[ix, iy] - failure_thickness
            
            # TODO: break failure volume up into a bunch of smaller parcels for serial routing
            # print(
            #     "sed parcel volume, and curent failure volume:",
            #     self.Vp_sed,
            #     failure_volume,
            # )

            fx, fy = int(ix), int(iy)  # define new coords for parcel steppping
            max_step = 25
            _continue = True
            _iter = 0
            while _continue:
                idir = dirmax[fx, fy]
                fx = fx + self.jwalk_flat[idir]
                fy = fy + self.iwalk_flat[idir]

                if self.cell_type[fx, fy] == -1:  # check for "edge" cell
                    _continue = False  # kill the `while` loop
                    # i.e., the sediment routes off the map
                    break

                eta_nbrs = pad_eta[fx - 1 + 1 : fx + 2 + 1, fy - 1 + 1 : fy + 2 + 1]
                potential_eta = self.eta[fx, fy] + failure_thickness
                local_slopes = np.tan(
                    (potential_eta - eta_nbrs.ravel()) / (distances * self.dx)
                )
                local_slope = np.max(local_slopes)
                
                ### to play around with. Failure code from Andrew. 
                # if local_slope < dep_slope_crit:
                #     # deposit
                #     # breakpoint()
                #     self.eta[fx, fy] = self.eta[fx, fy] + (
                #         failure_volume / self.dx / self.dx
                #     )
                #     _continue = False

                # else:
                #     # deposit the amount that lowers it below the threshold and then route the rest
                #     current_excess_height = np.tan(fail_slope_crit) * self.dx
                #     height_to_match_dep_slope_crit = np.tan(dep_slope_crit) * self.dx
                #     volume_to_deposit = (
                #         height_to_match_dep_slope_crit * self.dx * self.dx
                #     )
                #     if volume_to_deposit <= failure_volume:
                #         # if the failure has more volume than needed here
                #         self.eta[fx, fy] = self.eta[fx, fy] + (
                #             volume_to_deposit / self.dx / self.dx
                #         )
                #         failure_volume -= volume_to_deposit
                #         # take another step
                #     else:
                #         # only can deposit as much as remains in the failure volume
                #         volume_to_deposit = failure_volume
                #         self.eta[fx, fy] = self.eta[fx, fy] + (
                #             volume_to_deposit / self.dx / self.dx
                #         )
                #         _continue = False

                # deposit the amount that lowers it below the threshold and then route the rest
                height_to_match_dep_slope_crit = np.tan(dep_slope_crit) * self.dx
                volume_to_deposit = height_to_match_dep_slope_crit * self.dx * self.dx
                volume_to_deposit = np.minimum(volume_to_deposit, failure_volume)
                self.eta[fx, fy] = self.eta[fx, fy] + (
                    volume_to_deposit / self.dx / self.dx
                )
                failure_volume -= volume_to_deposit

                _iter += 1
                if failure_volume <= 0:
                    _continue = False
                if _iter == max_step:
                    _continue = False
        # save output of total failure volume during this timestep
        #self.total_failure_volume.append(total_failure_volume)
        pass