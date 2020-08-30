
import numpy as np
from numba import njit, jit, typed
from scipy import ndimage

from . import shared_tools

# tools for sediment routing algorithms and deposition/erosion


class sed_tools(object):

    def sed_route(self):
        """Sediment routing main method.

        This is the main method for sediment routing in the model. It is
        called once per `update()` call.
        """
        self.pad_depth = np.pad(self.depth, 1, 'edge')

        self.qs[:] = 0
        self.Vp_dep_sand[:] = 0
        self.Vp_dep_mud[:] = 0

        self.sand_route()

        self.mud_route()

        self.topo_diffusion()

    def deposit(self, Vp_dep, px, py):
        """Deposit sand or mud.

        Deposit sediment volume `Vp_dep`. The change in bed elevation depends
        on the sediment mass conservation (i.e., Exner equation) and in the
        case of deposition, is equal to:

        .. code::

            Vp_dep / (dx * dx)

        Following the sediment deposition, the new values for flow depth and
        flow velocity are determined. The `uw` field is updated in this
        routine, and the components of the flow velocity are then updated in
        the :obj:`update_u` method.

        Parameters
        ----------
        Vp_dep : :obj:`float`
            Volume of sediment to deposit.

        px : :obj:`int`
            Index.

        py : :obj:`int`
            Index.

        Returns
        -------
        """
        # Compute the changes in the bed elevation, and some flow fields
        #    The :obj:`_update_helper` function is a jitted routine which
        #    computes the new values for the bed elevation and then determines
        #    the updated values of the depth and flow velocity fields. These
        #    determinations require several comparisons and repeated indexing,
        #    so we use a jitted "helper" function to do the operations.
        eta, depth, uw = _update_helper(Vp_dep, self.dx, self.u_max,
                                        self.eta[px, py], self.depth[px, py],
                                        self.stage[px, py], self.qw[px, py],
                                        self.uw[px, py])

        # now apply the computed updated values
        self.eta[px, py] = eta  # update bed
        self.depth[px, py] = depth  # update depth
        self.pad_depth[px + 1, py + 1] = depth  # update depth in padded array
        self.uw[px, py] = uw  # update flow field
        self.update_u(px, py)  # update velocity fields

        self.Vp_res = self.Vp_res - Vp_dep  # update sed volume left in parcel

    def erode(self, Vp_ero, px, py):
        """Erode sand or mud.

        Erode sediment volume `Vp_ero`. The change in bed elevation depends
        on the sediment mass conservation (i.e., Exner equation) and in the
        case of erosion, is equal to:

        .. code::

            -Vp_ero / (dx * dx)

        Following the sediment erosion, the new values for flow depth and
        flow velocity are determined. The `uw` field is updated in this
        routine, and the components of the flow velocity are then updated in
        the :obj:`update_u` method.

        .. note::

            Total sediment mass is preserved, but individual categories
            of sand and mud are not. I.e., it is assumed that there is infinite
            sand and/or mud to erode at any location where erosion is slated to
            occur.

        Parameters
        ----------
        Vp_ero : :obj:`float`
            Volume of sediment to erode.

        px : :obj:`int`
            Index.

        py : :obj:`int`
            Index.

        Returns
        -------
        """
        # Compute the changes in the bed elevation, and some flow fields
        #    The :obj:`_update_helper` function is a jitted routine which
        #    computes the new values for the bed elevation and then determines
        #    the updated values of the depth and flow velocity fields. These
        #    determinations require several comparisons and repeated indexing,
        #    so we use a jitted "helper" function to do the operations.
        eta, depth, uw = _update_helper(-Vp_ero, self.dx, self.u_max,
                                        self.eta[px, py], self.depth[px, py],
                                        self.stage[px, py], self.qw[px, py],
                                        self.uw[px, py])

        # now apply the computed updated values
        self.eta[px, py] = eta  # update bed
        self.depth[px, py] = depth  # update depth
        self.pad_depth[px + 1, py + 1] = depth  # update depth in padded array
        self.uw[px, py] = uw  # update flow field
        self.update_u(px, py)  # update velocity fields

        self.Vp_res = self.Vp_res + Vp_ero

    def update_u(self, px, py):
        """Update velocity.

        Update velocities after erosion or deposition.
        """
        qw_loc = self.qw[px, py]
        uw_loc = self.uw[px, py]
        if qw_loc > 0:
            self.ux[px, py] = uw_loc * self.qx[px, py] / qw_loc
            self.uy[px, py] = uw_loc * self.qy[px, py] / qw_loc
        else:
            self.ux[px, py] = 0
            self.uy[px, py] = 0

    def sand_dep_ero(self, px, py):
        """Decide if erode or deposit sand.

        .. note:: Volumetric change is limited to 1/4 local cell water volume.
        """
        U_loc = self.uw[px, py]

        qs_cap = (self.qs0 * self.f_bedload / self.u0**self.beta *
                  U_loc**self.beta)

        qs_loc = self.qs[px, py]

        Vp_dep = 0
        Vp_ero = 0

        if qs_loc > qs_cap:
            # Sand deposition
            #     If more sediment is in transport than the determined
            #     transport capacity of the cell (`qs_cap`), sediment needs to
            #     deposit on the bed.
            Vp_dep = _regulate_Vp_change(self.Vp_res, self.stage[px, py],
                                         self.eta[px, py], self.dx)

            self.deposit(Vp_dep, px, py)

        elif (U_loc > self.U_ero_sand) and (qs_loc < qs_cap):
            # Sand erosion
            #     Can only occur if local velocity is greater than the
            #     critical erosion threshold for sand, *and* if the local
            #     transport capacity is not yet reached.
            Vp_ero = get_eroded_volume(self.Vp_sed, U_loc, self.U_ero_sand,
                                       self.beta, self.stage[px, py],
                                       self.eta[px, py], self.dx)
            self.erode(Vp_ero, px, py)

        self.Vp_dep_sand[px, py] = self.Vp_dep_sand[px, py] + Vp_dep

    def mud_dep_ero(self, px, py):
        """Decide if deposit or erode mud.

        .. note:: Volumetric change is limited to 1/4 local cell water volume.
        """
        U_loc = self.uw[px, py]

        Vp_dep = 0
        Vp_ero = 0

        if U_loc < self.U_dep_mud:

            Vp_dep = (self._lambda * self.Vp_res *
                      (self.U_dep_mud**self.beta - U_loc**self.beta) /
                      (self.U_dep_mud**self.beta))

            Vp_dep = _regulate_Vp_change(Vp_dep, self.stage[px, py],
                                        self.eta[px, py], self.dx)

            self.deposit(Vp_dep, px, py)

        if U_loc > self.U_ero_mud:

            Vp_ero = get_eroded_volume(self.Vp_sed, U_loc, self.U_ero_mud,
                                       self.beta, self.stage[px, py],
                                       self.eta[px, py], self.dx)

            self.erode(Vp_ero, px, py)

        self.Vp_dep_mud[px, py] = self.Vp_dep_mud[px, py] + Vp_dep

    def sed_parcel(self, theta_sed, sed, px, py):
        """Route one sediment parcel.

        This algorithm is called :obj:`~pyDeltaRCM.DeltaModel.Np_sed` times,
        and routes each parcel of sediment until the sediment volume has been
        depleted by deposition, or until the maximum number of steps for the
        parcel has been reached (`self.stepmax`).

        Algorithm is to:
          1. As input, receive the type of sediment (:obj:`sed_type`) and the
          starting location of the parcel as :obj:`px` and :obj:`py`.
          2. begin a while loop, which counts the number of steps the parcel
          has taken
          3. determine, based on the water surface and flow velocity field,
          which location to travel to next. This determination is wrapped into
          a jitted function :obj:`choose_next_sed_location`, which utilizes
          the `shared_tools` methods to pick the next location.
          4. call the appropriate deposition/erosion method for the
          `sed_type`, and proceed with deposition or erosion there. This step
          modifies the bed elevation and flow depth fields, which necessitates
          finding the new weights for routing on each step. Also, the volume
          of sediment is either increased or decreased (erosion or
          deposition).
          5. repeat from 3, until `stepmax` is reached, or the sediment parcel
          volume is depleted.

        .. note::

            We are unable to precompute the routing weights, in the way
            we do in :obj:`~pyDeltaRCM.water_tools.get_water_weight_array`.
        """
        it = 0
        sed_continue = 1

        while (it < self.stepmax):
            it += 1

            # Choose the next location for the parcel to travel to
            #     The location of the parcel is determined by ``(px, py)``,
            #     and we need to determine where the parcel should travel to
            #     next. The travel of the parcel depends primarily on the
            #     hydrailics of the delta, determined in the previous
            #     timestep. The following function is a jitted routine.
            weights, new_cell, dist, istep, jstep = choose_next_location(
                px, py, self.pad_stage, self.pad_depth, self.pad_cell_type,
                self.stage, self.qx, self.qy, self.ivec_flat, self.jvec_flat,
                self.distances_flat, self.dry_depth, self.gamma, theta_sed,
                self.iwalk_flat, self.jwalk_flat)

            # deposition and erosion
            if sed == 'sand':  # sand
                depoPart = self.Vp_res / 2 / self._dt / self.dx
                px, py, self.qs = shared_tools.partition_sand(
                    self.qs, depoPart, py, px, dist, istep, jstep)
                self.sand_dep_ero(px, py)

            if sed == 'mud':  # mud
                px = px + jstep
                py = py + istep
                self.mud_dep_ero(px, py)

            # check for "edge" cell
            if self.cell_type[px, py] == -1:
                it = float('inf')  # kill the `while` loop

    def sand_route(self):
        """Route sand parcels; topo diffusion."""
        theta_sed = self.theta_sand

        num_starts = int(self.Np_sed * self.f_bedload)
        inlet_weights = np.ones_like(self.inlet)
        start_indices = shared_tools.get_start_indices(self.inlet,
                                                       inlet_weights,
                                                       num_starts)

        for np_sed in range(num_starts):

            self.Vp_res = self.Vp_sed

            px = 0
            py = start_indices[np_sed]

            self.qs[px, py] = (self.qs[px, py] +
                               self.Vp_res / 2. / self._dt / self.dx)

            self.sed_parcel(theta_sed, 'sand', px, py)

    def topo_diffusion(self):
        """Diffuse topography after routing.

        Diffuse topography after routing all coarse sediment parcels. The
        method uses convolution with a kernel to compute smoother topography,
        and then adds this different to the current eta to do the smoothing.
        The operation is repeated `N_crossdiff` times.
        """
        for _ in range(self.N_crossdiff):

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
        """Route mud parcels."""
        theta_sed = self.theta_mud

        num_starts = int(self.Np_sed * (1 - self.f_bedload))
        inlet_weights = np.ones_like(self.inlet)
        start_indices = shared_tools.get_start_indices(self.inlet,
                                                       inlet_weights,
                                                       num_starts)

        for np_sed in range(num_starts):

            self.Vp_res = self.Vp_sed

            px = 0
            py = start_indices[np_sed]

            self.sed_parcel(theta_sed, 'mud', px, py)


@njit
def choose_next_location(px, py, pad_stage, pad_depth, pad_cell_type, stage,
                         qx, qy, ivec_flat, jvec_flat, distances_flat,
                         dry_depth, gamma, theta_sed, iwalk_flat, jwalk_flat):

    # choose next location with weights
    stage_nbrs = pad_stage[
        px - 1 + 1:px + 2 + 1, py - 1 + 1:py + 2 + 1]
    depth_nbrs = pad_depth[
        px - 1 + 1:px + 2 + 1, py - 1 + 1:py + 2 + 1]
    cell_type_ind = pad_cell_type[
        px - 1 + 1:px + 2 + 1, py - 1 + 1:py + 2 + 1]

    ind = (px, py)
    weight_sfc, weight_int = shared_tools.get_weight_sfc_int(
        stage[px, py], stage_nbrs.ravel(), qx[px, py],
        qy[px, py], ivec_flat, jvec_flat,
        distances_flat)
    weights = shared_tools.get_weight_at_cell(
        ind, weight_sfc, weight_int, depth_nbrs.ravel(),
        cell_type_ind.ravel(), dry_depth,
        gamma, theta_sed)
    new_cell = shared_tools.random_pick(weights)
    dist, istep, jstep, _ = shared_tools.get_steps(
        new_cell, iwalk_flat, jwalk_flat)

    return weights, new_cell, dist, istep, jstep


@njit
def get_eroded_volume(Vp_sed, U_loc, U_ero, beta,
                      stage, eta, dx):
    """Get volume of sediment eroded.
    """
    Vp_ero = _compute_Vp_ero(Vp_sed, U_loc, U_ero, beta)
    Vp_ero = _regulate_Vp_change(Vp_ero, stage,
                                eta, dx)
    return Vp_ero


@njit
def _compute_Vp_ero(Vp_sed, U_loc, U_ero, beta):
    """Compute volume erorded based on velocity.

    The volume of sediment eroded depends on the local flow velocity
    (:obj:`U_loc`), the critical velocity for erosion (:obj:`U_ero`), and
    `beta`.

    Function is used by both sand and mud pathways, but the input value of
    U_ero depends on the pathway.

    Parameters
    ----------
    Vp_sed : :obj:`float`
        Volume of a sediment parcel.

    U_loc : :obj:`float`
        Local flow velocity.

    U_ero : :obj:`float`
        Critical velocity for erosion.

    beta : :obj:`float`
        ?????

    Returns
    -------
    Vp_ero : :obj:`float`
        Volume of eroded sediment.
    """
    return (Vp_sed * (U_loc**beta - U_ero**beta) /
            U_ero**beta)


@njit
def _regulate_Vp_change(Vp, stage, eta, dx):
    """Limit change in volume to 1/4 of a cell volume.

    Function is used by multiple pathways in `mud_dep_ero` and `sand_dep_ero`
    but with different inputs for the sediment volume (:obj:`Vp`).
    """
    fourth = (stage - eta) / 4 * (dx*dx)
    return np.minimum(Vp, fourth)


@njit
def _update_helper(Vp_dep, dx, u_max, eta0, depth0, stage0, qw0, uw0):
    eta_change_loc = Vp_dep / (dx*dx)

    eta = eta0 + eta_change_loc  # new bed
    depth = stage0 - eta  # new depth
    depth = np.maximum(0, depth)  # force >=0

    if depth > 0:
        uw = np.minimum(u_max, qw0 / depth)
    else:
        uw = uw0

    return eta, depth, uw
