import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mpl_toolkits.axes_grid1 as axtk

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

        # overwrite the initial data with this new basin shape
        self.output_data()

    """
    def __init__(self, input_file, **kwargs):
        # inherit from base model
        super().__init__(input_file, **kwargs)
        #self.fail_deg_crit = self.fail_deg_crit
        #self.dep_deg_crit = self.dep_deg_crit
        # hook_after_create_domain is called here
        #fail_deg_crit = 30, dep_deg_crit = 14
        slope = 0.0005  # cross basin slope
        eta_line = slope * np.arange(0, self.length,
                                      step=self.dx)
        eta_grid = np.tile(eta_line, (self.L - self.L0, 1))
        eta_grid = eta_grid - ((slope * self.Width)/2)  # center at inlet
        self.eta[self.L0:, :] += eta_grid

        # overwrite the initial data with this new basin shape
        self.output_data()
"""
    def hook_topo_diffusion(self, **kwargs):
        if self._time_iter % 1 == 0:
            
            self.angle_of_repose()
            pass

    def angle_of_repose(self):
        """Routine for foreset processes with given dimensions."""
        """
        Code block 1, here we calculate the local slope maxima and their indices
        """
        # get distances from each cell and flattened into one dimesion
        distances = self.distances_flat
        #pad cells to add guarantee all cells have neighbors
        pad_eta = custom_pad(self.eta)
        #Construct an array with basin dimesions filled with vectors of 9 zeros
        slope = np.zeros((self.L, self.W, 9))
        #Calculate slope of each cell and surrounding neighbors and fill into vectors at each cells
        for i in np.arange(self.L):
            for j in np.arange(self.W):
                eta_nbrs = pad_eta[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]
                slope[i, j, :] = np.tan(
                    (self.eta[i, j] - eta_nbrs.ravel()) / (distances * self.dx)
                )  # units are radians
        #find local slope maxima in each set of surrounding cells
        dirmax = np.argmax(slope, axis=2)
        #find coordinates of local slope maxima
        m, n = dirmax.shape
        #
        I, J = np.ogrid[:m, :n]
        slopemax = slope[I, J, dirmax]

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
        eta_before = np.copy(self.eta)
        

        # parameters to be moved out to YAML config difference between these
        #   critical slope controls the size of the failure and therefore the
        #   coherence of the mass moving down the slope, and how far (indirectly) the mass moves before depositing.
        fail_deg_crit = 30  # critical slope for failure in degrees
        dep_deg_crit = 14  # critical slope for deposition in degrees
        #eventually have those as yaml parameters?
        fail_slope_crit = np.radians(fail_deg_crit)  # units as radians
        dep_slope_crit = np.radians(dep_deg_crit)

        steep = slopemax > fail_slope_crit
        cell = self.cell_type == 0
        whr_steep = np.where(np.logical_and(steep, cell))


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

            # print("steps taken:", _iter)

        # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        # ax[0].imshow(self.eta, cmap="cividis")
        # ax[1].imshow(slopemax)
        # ax[2].imshow(self.eta - eta_before, cmap="RdBu", vmin=-2, vmax=2)
        # plt.show(block=False)

        # breakpoint()
        # assert np.sum((self.eta - eta_before)) == 0

if __name__ == "__main__":
    # parameter choices for scaling
    If = 7 / 365.25  # intermittency factor for year-scaling

    # base yaml configuration
    base_output = "/Users/lucillebaker-stahl/Documents/pyDeltaRCM/aor_testing/aor_testing_outputs"
    base_yaml = "/Users/lucillebaker-stahl/Documents/pyDeltaRCM/pyDeltaRCM/default.yml"

    checkpoint_src = "./foreset_output_testing"
    # copy the spinup checkpoint to each of the folders
    # shutil.copy(
    #     src=os.path.join(checkpoint_src, "checkpoint.npz"),
    #     dst=os.path.join(base_output),
    # )

    _mdl = aor180Model(
        input_file=base_yaml,
        out_dir=base_output,
        save_checkpoint=True,
        resume_checkpoint=False,  # os.path.join(checkpoint_src),
        save_dt=864000,
        save_eta_figs=True,
        save_velocity_figs=False,
        clobber_netcdf=True,
    )

    # solve for how many timesteps
    # targ_dur = 1000  # target run duration (years)
    # targ_dur_mdl = (targ_dur * sec_in_day * day_in_yr) * If
    # tsteps = int((targ_dur_mdl // _mdl.time_step) + 1)

    tsteps = 1000

    for i in range(tsteps):
        _mdl.update()
    # try:
    #     for i in range(tsteps):
    #         _mdl.update()
    # except Exception as e:
    #     print("ERROR!")
    #     _mdl.logger.exception(e)

    # finalize
    _mdl.finalize()
