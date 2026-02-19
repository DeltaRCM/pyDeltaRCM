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


class ReservoirModel(pyDeltaRCM.DeltaModel):
    def __init__(self, input_file, **kwargs):
        # inherit from base model
        super().__init__(input_file, **kwargs)

        # hook_after_create_domain is called here

        # overwrite the initial data with this new basin shape
        self.output_data()

    def hook_topo_diffusion(self):
        if self._time_iter % 1 == 0:
            self.foreset_processes()
            pass

    def hook_after_create_domain(self):
        """Adjust the domain to be a reservoir delta."""

        # ---- domain ----
        cell_land = -2
        cell_channel = 1
        cell_ocean = 0
        cell_edge = -1

        initial_topography_flag = "simplfied"
        if initial_topography_flag == "dem":
            # load your initial topography DEM here.
            #   It will be simplest if you can rotate the DEM so that
            #   the inlet is along the first dimension of the array (i.e., along the top).
            #   The DEM will need to be a rectangle, and should be simplified
            #   to a resolution <10e3 grid cells. You'll need to set up the parameters
            #   in the yaml (e.g., `dx`, `Length`, `Width`) so that the DeltaModel is
            #   initialized with a domain that matches the size and resolution of
            #   the dem you're going to swap in.
            #
            #   After loading the dem, you'll need to manipulate the
            #   self.cell_type into something that matches the simple domain
            #   I've created below. E.g., you need to set th downstream end
            #   of the reservoir as -1, some reservoir valley walls as -2,
            #   and a few inlet channel cells as 1. Care will probably be needed to set up
            #   self.depth and self.stage to reasonable initial values given the
            #   new bed elevation.
            raise NotImplementedError()

        elif initial_topography_flag == "simplfied":
            # I'm setting up a simple rectangular grid here, to give a sense
            #   for the model working and setting the cell types.
            #   I do this by setting the water stage for every
            #   location in the domain.

            # define the stage from each side
            self.L0_s = self.L0 // 2  # L0, number of cells along each of the sides
            _top = np.maximum(0, self.L0 - self.y - 1) * self._dx * self._S0
            _left = np.maximum(0, self.L0_s - self.x - 1) * self._dx * self._S0
            _right = (
                np.maximum(0, self.L0_s - np.fliplr(self.x) - 1) * self._dx * self._S0
            )
            self.stage[:] = np.maximum(_top, np.maximum(_left, _right))
            self.stage[self.cell_type == cell_ocean] = 0.0

            # set the channel inlet location
            channel_inds = int(self.CTR - round(self.N0 / 2)) + 1
            y_channel_max = channel_inds + self.N0

            # set all of the ocean to ocean cell type
            self.stage[self.cell_type == cell_ocean] = 0.0

            # change the cell types along the edges to land
            self.cell_type = np.zeros((self.L, self.W), dtype=np.int64)
            self.cell_type[: self.L0, :] = cell_land
            self.cell_type[:, : self.L0_s] = cell_land
            self.cell_type[:, -self.L0_s :] = cell_land

            # block the corners to prevent channel trapping
            for i in range(2 * channel_inds + 1):
                for j in range(channel_inds - (i // 2) + 1):
                    self.cell_type[i, j] = cell_land
            for i in range((2 * (self.W - y_channel_max) + 2)):
                for j in range((self.W - y_channel_max + 2) - (i // 2)):
                    self.cell_type[i, -j] = cell_land

            # set the channel location
            self.cell_type[: self.L0, channel_inds:y_channel_max] = cell_channel

            # reset the depth
            self.depth = np.zeros((self.L, self.W), dtype=np.float32)
            # small taper between inlet and basin depth?

            # taper ocean deeper
            S0b = 0.075
            self.depth[:] = self.hb + (
                np.minimum(0, self.L0 - self.y - 1) * self._dx * (-S0b)
            )
            self.depth = np.minimum(100, self.depth)
            # fill in depth value at edges
            self.depth[self.cell_type == cell_channel] = self.h0
            self.depth[self.cell_type == cell_land] = 0

            # sp.ndimage.filters.convolve() np.full((3, 3, 3), 1.0/27)
            # smooth the walls
            is_water = self.cell_type == 0
            smoothed = sp.ndimage.uniform_filter(self.depth, size=5)
            self.depth[is_water] = smoothed[is_water]

            # set the model downstream edge to "edge" cell_type
            self.cell_type[-3:, :] = cell_edge

            # affirm the inlet conditions
            self.cell_type[: self.L0, :] = cell_land
            self.cell_type[: self.L0, channel_inds:y_channel_max] = cell_channel

            # update the bed elevation from stage and depth
            self.eta[:] = self.stage - self.depth

            # now update flow fields to good initial values
            self.qx[self.cell_type == cell_channel] = self.qw0
            self.qx[self.cell_type == cell_ocean] = self.qw0 / 10.0
            self.qw = (self.qx**2 + self.qy**2) ** (0.5)
            self.ux[self.depth > 0] = 0.2
            self.uy[self.depth > 0] = 0
            self.uw[self.depth > 0] = 0.2

        if False:  # make a plot to inpect the configuration
            etafig = self.make_figure("eta", 0)
            ctfig = self.make_figure("cell_type", 0)
            etafig.savefig("initial_eta.png")
            ctfig.savefig("initial_cell_type.png")
            # plt.close()
            plt.show()

        # reinitialize the sediment routers since
        #   we have changed some initial attributes
        self.init_sediment_routers()

    def foreset_processes(self):
        """Catch all routine for any foreset processes."""

        # get local slope angle map
        grad = np.gradient(self.eta, self.dx)

        distances = self.distances_flat
        pad_eta = custom_pad(self.eta)
        slope = np.zeros((self.L, self.W, 9))
        for i in np.arange(self.L):
            for j in np.arange(self.W):
                eta_nbrs = pad_eta[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]
                slope[i, j, :] = np.tan(
                    (self.eta[i, j] - eta_nbrs.ravel()) / (distances * self.dx)
                )  # units are radians

        dirmax = np.argmax(slope, axis=2)
        m, n = dirmax.shape
        I, J = np.ogrid[:m, :n]
        slopemax = slope[I, J, dirmax]

        """
        here, need to implement router
        basically:
          find places where the slope is steeper than some critical value
          randomly select them in some order
          calculate the thickness at that cell that is unstable (i.e., how much removed would lower the slope to below threshold)
          move that volume of sediment down the slope in the max direction
          in the future, could implement complex routing rules to approx diff turbidity currents?
        """
        # eta_before = np.copy(self.eta)
        eta_before = self.eta0

        # parameters to be moved out to YAML config difference between these
        #   critical slope controls the size of the failure and therefore the
        #   coherence of the mass moving down the slope, and how far (indirectly) the mass moves before depositing.
        fail_deg_crit = 30  # critical slope for failure in degrees
        dep_deg_crit = 14  # critical slope for deposition in degrees
        fail_slope_crit = np.radians(fail_deg_crit)  # units as radians
        dep_slope_crit = np.radians(dep_deg_crit)

        steep = slopemax > fail_slope_crit
        cell = self.cell_type == 0
        whr_steep = np.where(np.logical_and(steep, cell))
        for i, (ix, iy) in enumerate(zip(*whr_steep)):
            pad_eta = custom_pad(self.eta)  # recompute on each transport

            idir = dirmax[ix, iy]

            current_excess_height = np.tan(fail_slope_crit) * self.dx
            new_excess_height = np.tan(dep_slope_crit) * self.dx
            failure_thickness = current_excess_height - new_excess_height
            # TODO: limit failure thickness to initial bedrock
            # deposit_thickness = (
            #     self.eta[ix, iy] - self.eta0[ix, iy]
            # )  # could vectorize above loop

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

    def make_figure(self, var, timestep):
        """Custom figure for this domain shape.

        This overwrites the existing `make_figure` so that the long and skinny
        domain is easier to see.

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

        fig, ax = plt.subplots(figsize=(4, 8), dpi=200)
        im = ax.pcolormesh(self.X, self.Y, _data, shading="flat")
        ax.set_xlim((0, self._Width))
        ax.set_ylim((0, self._Length))
        ax.set_aspect("equal", adjustable="box")
        divider = axtk.axes_divider.make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7)
        ax.use_sticky_edges = False
        ax.margins(y=0.2)
        ax.set_title(str(var) + "\ntime: " + str(timestep), fontsize=10)

        return fig


if __name__ == "__main__":
    # parameter choices for scaling
    If = 7 / 365.25  # intermittency factor for year-scaling

    # base yaml configuration
    base_output = "foreset_output_testing"
    base_yaml = "./foreset.yaml"

    checkpoint_src = "./foreset_output_testing"
    # copy the spinup checkpoint to each of the folders
    # shutil.copy(
    #     src=os.path.join(checkpoint_src, "checkpoint.npz"),
    #     dst=os.path.join(base_output),
    # )

    _mdl = ReservoirModel(
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
