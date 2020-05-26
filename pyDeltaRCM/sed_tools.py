
import sys
import os
import re
import string
import logging
import time

from math import floor, sqrt, pi
import numpy as np
from random import shuffle

import matplotlib
from matplotlib import pyplot as plt

from scipy.sparse import lil_matrix, csc_matrix, hstack
from scipy import ndimage

from netCDF4 import Dataset

from . import shared_tools

# tools for sediment routing algorithms and deposition/erosion


class sed_tools(object):

    def sed_route(self):
        """route all sediment"""

        self.pad_depth = np.pad(self.depth, 1, 'edge')

        self.qs[:] = 0
        self.Vp_dep_sand[:] = 0
        self.Vp_dep_mud[:] = 0

        self.sand_route()

        self.mud_route()

        self.topo_diffusion()

    def deposit(self, Vp_dep, px, py):
        """deposit sand or mud"""

        eta_change_loc = Vp_dep / self.dx**2

        self.eta[px, py] = self.eta[px, py] + eta_change_loc  # update bed
        self.depth[px, py] = self.stage[px, py] - self.eta[px, py]

        if self.depth[px, py] < 0:
            self.depth[px, py] = 0

        self.pad_depth[px + 1, py + 1] = self.depth[px, py]

        if self.depth[px, py] > 0:
            self.uw[px, py] = min(self.u_max, self.qw[
                                  px, py] / self.depth[px, py])

        self.update_u(px, py)

        self.Vp_res = self.Vp_res - Vp_dep
        # update amount of sediment left in parcel

    def erode(self, Vp_ero, px, py):
        """erode sand or mud
        total sediment mass is preserved but individual categories
        of sand and mud are not"""

        eta_change_loc = -Vp_ero / (self.dx**2)

        self.eta[px, py] = self.eta[px, py] + eta_change_loc
        self.depth[px, py] = self.stage[px, py] - self.eta[px, py]

        if self.depth[px, py] < 0:
            self.depth[px, py] = 0

        self.pad_depth[px + 1, py + 1] = self.depth[px, py]

        if self.depth[px, py] > 0:
            self.uw[px, py] = min(self.u_max, self.qw[
                                  px, py] / self.depth[px, py])

        self.update_u(px, py)

        self.Vp_res = self.Vp_res + Vp_ero

    def update_u(self, px, py):
        """update velocities after erosion or deposition"""

        if self.qw[px, py] > 0:

            self.ux[px, py] = self.uw[px, py] * \
                self.qx[px, py] / self.qw[px, py]
            self.uy[px, py] = self.uw[px, py] * \
                self.qy[px, py] / self.qw[px, py]

        else:
            self.ux[px, py] = 0
            self.uy[px, py] = 0

    def sand_dep_ero(self, px, py):
        """decide if erode or deposit sand"""

        U_loc = self.uw[px, py]

        qs_cap = (self.qs0 * self.f_bedload / self.u0**self.beta *
                  U_loc**self.beta)

        qs_loc = self.qs[px, py]

        Vp_dep = 0
        Vp_ero = 0

        if qs_loc > qs_cap:
            # if more sed than transport capacity has gone through cell
            # deposit sand

            Vp_dep = min(self.Vp_res,
                         (self.stage[px, py] - self.eta[px, py]) / 4. *
                         (self.dx**2))

            self.deposit(Vp_dep, px, py)

        elif (U_loc > self.U_ero_sand) and (qs_loc < qs_cap):
            # erosion can only occur if haven't reached transport capacity

            Vp_ero = (self.Vp_sed *
                      (U_loc**self.beta - self.U_ero_sand**self.beta) /
                      self.U_ero_sand**self.beta)

            Vp_ero = min(Vp_ero,
                         (self.stage[px, py] - self.eta[px, py]) / 4. *
                         (self.dx**2))

            self.erode(Vp_ero, px, py)

        self.Vp_dep_sand[px, py] = self.Vp_dep_sand[px, py] + Vp_dep

    def mud_dep_ero(self, px, py):
        """decide if deposit or erode mud"""

        U_loc = self.uw[px, py]

        Vp_dep = 0
        Vp_ero = 0

        if U_loc < self.U_dep_mud:

            Vp_dep = (self._lambda * self.Vp_res *
                      (self.U_dep_mud**self.beta - U_loc**self.beta) /
                      (self.U_dep_mud**self.beta))

            Vp_dep = min(Vp_dep,
                         (self.stage[px, py] - self.eta[px, py]) / 4. *
                         (self.dx**2))
            # change limited to 1/4 local depth

            self.deposit(Vp_dep, px, py)

        if U_loc > self.U_ero_mud:

            Vp_ero = (self.Vp_sed *
                      (U_loc**self.beta - self.U_ero_mud**self.beta) /
                      self.U_ero_mud**self.beta)

            Vp_ero = min(Vp_ero,
                         (self.stage[px, py] - self.eta[px, py]) / 4. *
                         (self.dx**2))
            # change limited to 1/4 local depth

            self.erode(Vp_ero, px, py)

        self.Vp_dep_mud[px, py] = self.Vp_dep_mud[px, py] + Vp_dep

    def sed_parcel(self, theta_sed, sed, px, py):
        """route one sediment parcel"""

        it = 0
        sed_continue = 1

        while (sed_continue == 1) and (it < self.itmax):
            # choose next with weights

            it += 1
            depth_ind = self.pad_depth[
                px - 1 + 1:px + 2 + 1, py - 1 + 1:py + 2 + 1]
            cell_type_ind = self.pad_cell_type[
                px - 1 + 1:px + 2 + 1, py - 1 + 1:py + 2 + 1]

            w1 = shared_tools.get_flux_wt(self.qx[(px, py)], self.qy[(px, py)], self.ivec, self.jvec)
            w2 = shared_tools.get_depth_wt(depth_ind, theta_sed)

            w3 = shared_tools.get_combined_weight(w1, w2, self.distances)

            weights = shared_tools.get_filtered_weight(
                w3.flatten(), px, depth_ind.flatten(),
                cell_type_ind.flatten(), self.dry_depth)

            new_cell = shared_tools.random_pick(weights)

            dist, istep, jstep, _ = shared_tools.get_steps(new_cell, self.iwalk.flat[:], self.jwalk.flat[:])

            # deposition and erosion

            if sed == 'sand':  # sand

                depoPart = self.Vp_res / 2 / self.dt / self.dx

                px, py, self.qs = shared_tools.partition_sand(self.qs, depoPart, py, px, dist, istep, jstep)

                self.sand_dep_ero(px, py)

            if sed == 'mud':  # mud

                px = px + jstep
                py = py + istep

                self.mud_dep_ero(px, py)

            if self.cell_type[px, py] == -1:
                sed_continue = 0

    def sand_route(self):
        """route sand parcels; topo diffusion"""

        theta_sed = self.theta_sand

        num_starts = int(self.Np_sed * self.f_bedload)
        inlet_weights = np.ones_like(self.inlet)
        start_indices = [
            self.inlet[shared_tools.random_pick(inlet_weights / sum(inlet_weights))]
            for x in range(num_starts)]

        for np_sed in range(num_starts):

            self.Vp_res = self.Vp_sed

            px = 0
            py = start_indices[np_sed]

            self.qs[px, py] = (self.qs[px, py] +
                               self.Vp_res / 2. / self.dt / self.dx)

            self.sed_parcel(theta_sed, 'sand', px, py)

    def topo_diffusion(self):
        """
        Diffuse topography after routing all coarse sediment parcels
        """

        for crossdiff in range(self.N_crossdiff):

            a = ndimage.convolve(self.eta, self.kernel1, mode='constant')
            b = ndimage.convolve(self.qs, self.kernel2, mode='constant')
            c = ndimage.convolve(self.qs * self.eta, self.kernel2,
                                 mode='constant')

            self.cf = (self.diffusion_multiplier *
                       (self.qs * a - self.eta * b + c))

            self.cf[self.cell_type == -2] = 0
            self.cf[0, :] = 0

            self.eta += self.cf

    def mud_route(self):
        """route mud parcels"""

        theta_sed = self.theta_mud

        num_starts = int(self.Np_sed * (1 - self.f_bedload))
        inlet_weights = np.ones_like(self.inlet)
        start_indices = [
            self.inlet[shared_tools.random_pick(inlet_weights / sum(inlet_weights))]
            for x in range(num_starts)]

        for np_sed in range(num_starts):

            self.Vp_res = self.Vp_sed

            px = 0
            py = start_indices[np_sed]

            self.sed_parcel(theta_sed, 'mud', px, py)
