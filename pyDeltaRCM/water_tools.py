
import numpy as np
from numba import njit, prange
import abc

from . import shared_tools

import matplotlib.pyplot as plt

# tools for water routing algorithms


class water_tools(abc.ABC):

    def init_water_iteration(self):
        _msg = 'Initializing water iteration'
        self.log_info(_msg, verbosity=2)

        self.qxn[:] = 0
        self.qyn[:] = 0
        self.qwn[:] = 0

        self.free_surf_flag[:] = 1  # all parcels begin as valid
        self.free_surf_walk_indices[:] = 0
        self.sfc_visit[:] = 0
        self.sfc_sum[:] = 0

        self.pad_stage = np.pad(self.stage, 1, 'edge')
        self.pad_depth = np.pad(self.depth, 1, 'edge')
        self.pad_cell_type = np.pad(self.cell_type, 1, 'edge')

    def run_water_iteration(self):
        """Run a single iteration of travel paths for all water parcels.

        Runs all water  parcels (`Np_water` parcels) for `stepmax` steps, or
        until the parcels reach a boundary.

        All parcels are processed in parallel, taking one step for each loop
        of the ``while`` loop.
        """
        _msg = 'Beginning stepping of water parcels'
        self.log_info(_msg, verbosity=2)

        _step = 0  # the step number of parcels
        inlet_weights = np.ones_like(self.inlet)
        start_indices = shared_tools.get_start_indices(self.inlet,
                                                       inlet_weights,
                                                       self._Np_water)

        # reset the free surface flag on each iteration 
        self.free_surf_flag[:] = 1

        # flux from ghost node
        self.qxn.flat[start_indices] += 1
        self.qyn.flat[start_indices] += 0  # this could be omitted...
        self.qwn.flat[start_indices] += self.Qp_water / self._dx / 2

        # load the initial indices into the walk indices
        self.free_surf_walk_indices[:, _step] = start_indices
        current_inds = np.copy(start_indices)

        self.get_water_weight_array()
        water_weights_flat = self.water_weights.reshape(-1, 9)  # flatten for fast access

        # while any parcles still need to take any steps
        while (sum(current_inds) > 0) & (_step < self.stepmax):

            # step counter increment
            _step += 1

            # check whether storage size needs to be increased
            self.check_size_of_indices_matrix(_step)

            # use water weights and random pick to determine d8 direction
            new_direction = _choose_next_direction(current_inds, water_weights_flat)
            new_direction = new_direction.astype(np.int)

            # use the new directions for each parcel to determine the new ind for each parcel
            new_inds = _calculate_new_ind(
                current_inds,
                new_direction,
                self.iwalk_flat,
                self.jwalk_flat,
                self.eta.shape)

            # determine the step of each parcel made based on the new direction
            dist, istep, jstep, astep = shared_tools.get_steps(
                new_direction,
                self.iwalk_flat,
                self.jwalk_flat)

            # breakpoint()

            # update the discharge field with walk of parcels
            #    this updates flux out of the old inds, and into the new inds
            self.update_Q(
                dist, current_inds, new_inds, astep, jstep, istep,
                update_current=True, update_next=True)

            # check for any loops in the walks of parcels
            #     A loop disqualifies a parcel from the free surface
            #     calculation, but does not stop the parcel from routing, and
            #     thus influencing the qw fields. If a parcel is looped, it
            #     will be given a new location in the domain, along the
            #     mean-transport vector.
            new_inds, looped = _check_for_loops(
                self.free_surf_walk_indices, new_inds, _step, self.L0, 
                self.eta.shape, self.CTR)
            looped = looped.astype(np.bool)

            # set the current_inds to be the new_inds values (i.e., take the step)
            current_inds[:] = new_inds[:]

            # invalidate the looped parcels from the free surface
            # breakpoint()
            self.free_surf_flag[looped] = 0  

            # APPLY the current_inds to be the new_inds values (i.e., take the step)
            # current_inds[:] = new_inds[:]

            # check for parcels that have reached the boundary
            boundary = self.check_for_boundary(current_inds)  # changes `free_surf_flag`

            # parcels that have looped are 
            boundary_looped = np.logical_and(looped, boundary)
            current_inds[boundary_looped] = 0

            # Record the parcel pathways for computing the free surface later
            self.free_surf_walk_indices[:, _step] = current_inds  # record indices

            # parcels that have reached the boundary are set ``ind==0``,
            #     effectively ending the routing of these parcels.
            current_inds[boundary] = 0


            # update the q*n fields for the final step 
            #    to ensure flux balanced at domain edge
            # if np.any(boundary):
            #     self.update_Q(
            #         dist[boundary], current_inds[boundary], 
            #         new_inds[boundary], astep[boundary],
            #         jstep[boundary], istep[boundary],
            #         update_current=True, update_next=False)

    def compute_free_surface(self):
        """Calculate free surface after routing all water parcels.

        This method uses the `free_surf_walk_indices` matrix accumulated
        during the routing of the water parcels (in
        :obj:`run_water_iteration`) to determine the free surface. The
        operations of the free surface computation are placed in a jitted
        function :obj:`accumulate_free_surface_walks`. Following this
        computation, the free surface is smoothed by steps in
        :obj:`finalize_free_surface`.
        """
        _msg = 'Computing free surface from water parcels'
        self.log_info(_msg, verbosity=2)

        self.sfc_visit, self.sfc_sum = _accumulate_free_surface_walks(
            self.free_surf_walk_indices, self.free_surf_flag, self.cell_type,
            self.uw, self.ux, self.uy, self.depth,
            self._dx, self._u0, self.h0, self._H_SL, self._S0)

        self.finalize_free_surface()

    def finalize_water_iteration(self, iteration):
        """Finish updating flow fields.

        Clean up at end of water iteration
        """
        _msg = 'Finalizing stepping of water parcels'
        self.log_info(_msg, verbosity=2)

        # apply an update on the depth to match stage and eta values
        self.depth[:] = np.maximum(self.stage - self.eta, 0)  # never negative

        # update fields for sediment routing
        self.update_flow_field(iteration)
        self.update_velocity_field()

    def check_size_of_indices_matrix(self, it):
        """Check if step path matrix needs to be made larger.

        Initial size of self.free_surf_walk_indices is half of self.stepmax
        because the number of iterations doesn't go beyond
        that for many timesteps.

        Once it reaches it > self.stepmax/2 once, make the size
        self.iter for all further timesteps
        """
        if it >= self.free_surf_walk_indices.shape[1]:
            _msg = 'Increasing size of self.free_surf_walk_indices'
            self.log_info(_msg, verbosity=2)
            
            indices_blank = np.zeros(
                (np.int(self._Np_water), np.int(self.stepmax / 4)), dtype=np.int)

            self.free_surf_walk_indices = np.hstack((self.free_surf_walk_indices, indices_blank))

    def get_water_weight_array(self):
        """Get step direction weights for each cell.

        This method is called once, before parcels are stepped, because the
        weights do not change during the stepping of parcels.
        """
        _msg = 'Computing water weight array'
        self.log_info(_msg, verbosity=2)
        self.water_weights = np.zeros(shape=(self.L, self.W, 9))

        for i in range(self.L):
            for j in range(self.W):
                stage_nbrs = self.pad_stage[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                depth_nbrs = self.pad_depth[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                ct_nbrs = self.pad_cell_type[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]

                weight_sfc, weight_int = shared_tools.get_weight_sfc_int(
                    self.stage[i, j], stage_nbrs.ravel(),
                    self.qx[i, j], self.qy[i, j], self.ivec_flat, self.jvec_flat,
                    self.distances_flat)

                self.water_weights[i, j] = _get_weight_at_cell_water(
                    (i, j), weight_sfc, weight_int,
                    depth_nbrs.ravel(), ct_nbrs.ravel(),
                    self.dry_depth, self.gamma, self._theta_water)

    def update_Q(self, dist, current_inds, next_inds, astep, jstep, istep,
                 update_current=False, update_next=False):
        """Update discharge field values after one set of water parcel steps."""
        _msg = 'Updating flux fields after single parcel step'
        self.log_info(_msg, verbosity=2)
        if update_current:
            self.qxn = _update_dirQfield(
                self.qxn.flat[:], dist, current_inds,
                astep, jstep).reshape(self.qxn.shape)
            self.qyn = _update_dirQfield(
                self.qyn.flat[:], dist, current_inds,
                astep, istep).reshape(self.qyn.shape)
            self.qwn = _update_absQfield(
                self.qwn.flat[:], dist, current_inds,
                astep, self.Qp_water, self._dx).reshape(self.qwn.shape)
        if update_next:
            self.qxn = _update_dirQfield(
                self.qxn.flat[:], dist, next_inds,
                astep, jstep).reshape(self.qxn.shape)
            self.qyn = _update_dirQfield(
                self.qyn.flat[:], dist, next_inds,
                astep, istep).reshape(self.qyn.shape)
            self.qwn = _update_absQfield(
                self.qwn.flat[:], dist, next_inds,
                astep, self.Qp_water, self._dx).reshape(self.qwn.shape)

    def check_for_boundary(self, inds):
        """Check whether parcels have reached the boundary.

        Checks whether any parcels have reached the model boundaries. If they
        have, then update the information in
        :obj:`~pyDeltaRCM.DeltaModel.free_surf_flag`.

        Parameters
        ----------
        inds : :obj:`ndarray`
            Unraveled indicies of parcels.
        """
        _msg = 'Checking stepped parcels against boundary location'
        self.log_info(_msg, verbosity=2)

        # where cell type is "edge" and free_surf_flag is currently valid (value: 0)
        # self.free_surf_flag[(self.cell_type.flat[inds] == -1) & (self.free_surf_flag == 0)] = 1

        # where cell type is "edge" and free_surf_flag is currently looped (value: -1)
        # self.free_surf_flag[(self.cell_type.flat[inds] == -1) & (self.free_surf_flag == -1)] = 2

        # inds[self.free_surf_flag == 2] = 0
        boundary = (self.cell_type.flat[inds] == -1)
        return boundary

    def finalize_free_surface(self):
        """Finalize the water free surface.

        This method occurs after the initial computation of the free surface,
        by accumulating the directed walks of all water parcels. In this
        method, thresholding is applied to correct for sea level, and a the
        free surface is smoothed by a jitted function (:obj:`_smooth_free_surface`).
        """
        _msg = 'Smoothing and finalizing free surface'
        self.log_info(_msg, verbosity=2)

        # begin from the previous stage
        Hnew = self.eta + self.depth

        # water surface height not under sea level
        Hnew[Hnew < self._H_SL] = self._H_SL

        # find average water surface elevation for a cell from accumulation
        Hnew[self.sfc_visit > 0] = (self.sfc_sum[self.sfc_visit > 0] /
                                    self.sfc_visit[self.sfc_visit > 0])

        # smooth newly calculated free surface
        Hsmth = _smooth_free_surface(
            Hnew, self.cell_type, self._Nsmooth, self._Csmooth)

        # combine new smoothed and previous free surf with underrelaxation
        if self._time_iter > 0:
            self.stage = (((1 - self._omega_sfc) * self.stage) + 
                          (self._omega_sfc * Hsmth))

        # apply a flooding correction
        self.flooding_correction()

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(self.sfc_visit)
        # ax[1].imshow(self.stage)
        # plt.show()

    def update_flow_field(self, iteration):
        """Update water discharge.

        Update the water discharge field after one set of water parcels
        iteration.
        """
        _msg = 'Updating discharge fields after parcel stepping'
        self.log_info(_msg, verbosity=2)

        dloc = (self.qxn**2 + self.qyn**2)**(0.5)

        qwn_div = np.ones((self.L, self.W))
        qwn_div[dloc > 0] = self.qwn[dloc > 0] / dloc[dloc > 0]

        self.qxn *= qwn_div
        self.qyn *= qwn_div

        if self._time_iter > 0:

            omega = self.omega_flow_iter
            if iteration == 0:
                omega = self._omega_flow

            self.qx = self.qxn * omega + self.qx * (1 - omega)
            self.qy = self.qyn * omega + self.qy * (1 - omega)

        else:
            self.qx = self.qxn.copy()
            self.qy = self.qyn.copy()

        self.qw = (self.qx**2 + self.qy**2)**(0.5)

        self.qx[0, self.inlet] = self.qw0
        self.qy[0, self.inlet] = 0
        self.qw[0, self.inlet] = self.qw0

    def update_velocity_field(self):
        """Update flow velocity fields.

        Update the flow velocity fields after one set of water parcels
        iteration.
        """
        _msg = 'Updating flow velocity fields after parcel stepping'
        self.log_info(_msg, verbosity=2)
        
        mask = np.logical_and((self.depth > self.dry_depth), (self.qw > 0))

        self.uw[mask] = np.minimum(self.u_max,
                                   self.qw[mask] / self.depth[mask])
        self.ux[mask] = self.uw[mask] * self.qx[mask] / self.qw[mask]
        self.uy[mask] = self.uw[mask] * self.qy[mask] / self.qw[mask]

        self.uw[~mask] = 0
        self.ux[~mask] = 0
        self.uy[~mask] = 0

    def flooding_correction(self):
        """Flood dry cells along the shore if necessary.

        Check the neighbors of all dry cells. If any dry cells have wet
        neighbors, check that their stage is not higher than the bed elevation
        of the center cell.
        If it is, flood the dry cell.
        """
        _msg = 'Computing flooding correction'
        self.log_info(_msg, verbosity=2)

        wet_mask = self.depth > self.dry_depth
        wet_mask_nh = self.get_wet_mask_nh()
        wet_mask_nh_sum = np.sum(wet_mask_nh, axis=0)

        # makes wet cells look like they have only dry neighbors
        wet_mask_nh_sum[wet_mask] = 0

        # indices of dry cells with wet neighbors
        shore_ind = np.where(wet_mask_nh_sum > 0)

        stage_nhs = self.build_weight_array(self.stage)
        eta_shore = self.eta[shore_ind]

        for i in range(len(shore_ind[0])):

            # pretends dry neighbor cells have stage zero
            #    so they cannot be > eta_shore[i]
            stage_nh = wet_mask_nh[:, shore_ind[0][i], shore_ind[1][i]] * \
                stage_nhs[:, shore_ind[0][i], shore_ind[1][i]]

            if (stage_nh > eta_shore[i]).any():
                self.stage[shore_ind[0][i], shore_ind[1][i]] = max(stage_nh)

    def get_wet_mask_nh(self):
        """Get wet mask.

        Returns np.array((8,L,W)), for each neighbor around a cell
        with 1 if the neighbor is wet and 0 if dry
        """
        wet_mask = (self.depth > self.dry_depth) * 1
        wet_mask_nh = self.build_weight_array(wet_mask, fix_edges=True)

        return wet_mask_nh

    def build_weight_array(self, array, fix_edges=False, normalize=False):
        """Weighting array of neighbors.

        Create np.array((8,L,W)) of quantity a
        in each of the neighbors to a cell
        """
        a_shape = array.shape

        wgt_array = np.zeros((8, a_shape[0], a_shape[1]))
        nums = list(range(8))

        wgt_array[nums[0], :, :-1] = array[:, 1:]  # E
        wgt_array[nums[1], 1:, :-1] = array[:-1, 1:]  # NE
        wgt_array[nums[2], 1:, :] = array[:-1, :]  # N
        wgt_array[nums[3], 1:, 1:] = array[:-1, :-1]  # NW
        wgt_array[nums[4], :, 1:] = array[:, :-1]  # W
        wgt_array[nums[5], :-1, 1:] = array[1:, :-1]  # SW
        wgt_array[nums[6], :-1, :] = array[1:, :]  # S
        wgt_array[nums[7], :-1, :-1] = array[1:, 1:]  # SE

        if fix_edges:
            wgt_array[nums[0], :, -1] = wgt_array[nums[0], :, -2]
            wgt_array[nums[1], :, -1] = wgt_array[nums[1], :, -2]
            wgt_array[nums[7], :, -1] = wgt_array[nums[7], :, -2]
            wgt_array[nums[1], 0, :] = wgt_array[nums[1], 1, :]
            wgt_array[nums[2], 0, :] = wgt_array[nums[2], 1, :]
            wgt_array[nums[3], 0, :] = wgt_array[nums[3], 1, :]
            wgt_array[nums[3], :, 0] = wgt_array[nums[3], :, 1]
            wgt_array[nums[4], :, 0] = wgt_array[nums[4], :, 1]
            wgt_array[nums[5], :, 0] = wgt_array[nums[5], :, 1]
            wgt_array[nums[5], -1, :] = wgt_array[nums[5], -2, :]
            wgt_array[nums[6], -1, :] = wgt_array[nums[6], -2, :]
            wgt_array[nums[7], -1, :] = wgt_array[nums[7], -2, :]

        if normalize:
            a_sum = np.sum(wgt_array, axis=0)
            wgt_array[:, a_sum != 0] = wgt_array[
                :, a_sum != 0] / a_sum[a_sum != 0]

        return wgt_array


@njit
def _get_weight_at_cell_water(ind, weight_sfc, weight_int, depth_nbrs, ct_nbrs,
                              dry_depth, gamma, theta):

    # create a fixed set of bools
    dry = (depth_nbrs <= dry_depth)
    wall = (ct_nbrs == -2)
    ctr = (np.arange(9) == 4)
    drywall = np.logical_or(dry, wall)
    invalid = np.logical_or(wall, ctr)
    if ind[0] == 0:
        invalid[:3] = True

    # set drywall to zero
    weight_sfc[drywall] = 0
    weight_int[drywall] = 0

    # set ctr to 0 before rebalancing
    weight_sfc[ctr] = 0
    weight_int[ctr] = 0

    # initial rebalance weights
    if np.nansum(weight_sfc) > 0:
        weight_sfc = weight_sfc / np.nansum(weight_sfc)
    if np.nansum(weight_int) > 0:
        weight_int = weight_int / np.nansum(weight_int)

    weight = gamma * weight_sfc + (1 - gamma) * weight_int
    weight[~dry] = (depth_nbrs[~dry] ** theta) * weight[~dry]

    # enforce disallowed choice to not move
    weight[ctr] = 0

    # sanity check
    nanweight = np.isnan(weight)
    if np.any(nanweight):
        raise RuntimeError('NaN encountered in water weighting. '
                           'Please report error.')

    # correct the weights for random choice 
    if np.nansum(weight) > 0:
        # if any cells have positive weights, rebalance to sum() == 1
        weight = weight / np.nansum(weight)
        weight[nanweight] = 0

    elif np.nansum(weight) == 0:
        # if all weights are zeros/nan
        if wall[4] == True: # np.all(wall):
            # if current cell is wall cell
            #    just set to zero and pass
            weight[:] = 0
        else:
            # convert to random walk into...
            wetvalid = np.logical_and(~dry, ~invalid)
            nwetvalid = np.nansum(wetvalid)
            if nwetvalid > 0:
                # any wet (and valid)
                weight[:] = 0
                weight[wetvalid] = (1 / nwetvalid)
            else:
                # no wet (and valid) neighbors. How did we get here?? Flooding corrections?
                #    allow to walk into any non-land cell, true random
                nnotwall = np.nansum(~wall)
                weight[:] = 0
                if nnotwall > 0:
                    weight[~wall] = (1 / nnotwall)
                else:
                    raise RuntimeError('No non-wall cells surrounding cell. '
                                       'Please report error.')

            weight = weight / np.nansum(weight)

    else:
        raise RuntimeError('Water sum(weight) less than 0. '
                           'Please report error.')

    return weight


@njit('int64[:](int64[:], float64[:,:])')
def _choose_next_direction(inds, water_weights):
    """Get new cell locations, based on water weights.

    Algorithm is to:
        1. loop through each parcel, which is described by a pair in the
           `inds` array.
        2. determine the water weights for that location (from `water_weights`)
        3. choose a new cell based on the probabilities of the weights (using
           the `random_pick` function)

    Parameters
    ----------
    inds : :obj:`ndarray`
        Current unraveled indices of the parcels. ``(N,)``  `ndarray`
        containing the unraveled indices.

    water_weights : :obj:`ndarray`
        Weights of every water cell. ``(LxW, 9)`` `ndarray`, uses unraveled
        indicies along 0th dimension; 9 cells represent self and 8 neighboring
        cells.

    Returns
    -------
    new_cells : :obj:`ndarray`
        The new cell for water parcels, relative to the current location.
        I.e., this is the D8 direction the parcel is going to travel in the
        next stage, :obj:`pyDeltaRCM.shared_tools._calculate_new_ind`.
    """
    new_cells = []
    for i in np.arange(inds.shape[0]):
        ind = inds[i]
        if ind != 0:
            weight = water_weights[ind, :]
            new_cells.append(shared_tools.random_pick(weight))
        else:
            new_cells.append(4)

    new_cells = np.array(new_cells)
    return new_cells


@njit
def _calculate_new_ind(indices, new_cells, iwalk, jwalk, domain_shape):
    """Calculate the new location (indices) of parcels.

    Use the information of the current parcel (`indices`) in conjunction with
    the D8 direction the parcel needs to travel (`new_cells`) to determine the
    new indices of each parcel.
    """
    newbies = []
    for p, q in zip(indices, new_cells):
        if q != 4:
            ind_tuple = shared_tools.custom_unravel(p, domain_shape)
            new_ind = (ind_tuple[0] + jwalk[q],
                       ind_tuple[1] + iwalk[q])
            newbies.append(shared_tools.custom_ravel(new_ind, domain_shape))
        else:
            newbies.append(0)

    return np.array(newbies)


@njit
def _check_for_loops(free_surf_walk_inds, new_inds, _step,
                     L0, domain_shape, CTR):
    """Check for loops in water parcel pathways.

    Look for looping random walks, i.e., where a parcel returns to somewhere
    it has already been.

    Parameters
    ----------


    Returns
    -------
    new_inds
        An updated array of parcel indicies, where the index of a parcel has
        been changed, if and only if, that parcel was looped.

    looped
        A binary integer array indicating whether a parcel was determined to
        have been looped.
    """
    nparcels = free_surf_walk_inds.shape[0]
    domain_min_x = domain_shape[0] - 2
    domain_min_y = domain_shape[1] - 2

    looped = np.zeros_like(new_inds)

    # if the _step number is larger than the inlet length
    if (_step > L0):
        # loop though every parcel walk
        for p in np.arange(nparcels):
            new_ind = new_inds[p]  # the new index of the parcel
            walk = free_surf_walk_inds[p, :]  # the parcel's walk
            _walk = walk[walk > 0]
            if (new_ind > 0):
                has_repeat_ind = len(_walk) != len(set(_walk))
                if has_repeat_ind:
                    # handle when a loop is detected
                    px, py = shared_tools.custom_unravel(new_ind, domain_shape)

                    # compute a new location for the parcel along the
                    #     mean-transport vector
                    Fx = px - 1
                    Fy = py - CTR
                    Fw = np.sqrt(Fx**2 + Fy**2)
                    if Fw != 0:
                        px = px + int(np.round(Fx / Fw * 5.))
                        py = py + int(np.round(Fy / Fw * 5.))

                    # limit the new px and py to beyond the inlet, and
                    #     away from domain edges
                    px = np.minimum(domain_min_x, np.maximum(px, L0))
                    py = np.minimum(domain_min_y, np.maximum(1, py))

                    nind = shared_tools.custom_ravel((px, py), domain_shape)
                    new_inds[p] = nind
                    looped[p] = 1  # this parcel is looped

    return new_inds, looped # free_surf_flag

"""Something is wrong with the check for loops.
First: it results in few walks contributing to the free surface (usually ~50%).
I believe this may contribute to the lower-than-levee stages in channels. 

One issue I see right away is that, although the check only looks for loops if
the step is beyond L0, it will trigger on any loop that is within the L0 too.
This NEEDS to be fixed. Hopefully that will help.
"""




@njit
def _update_dirQfield(qfield, dist, inds, astep, dirstep):
    """Update unit vector of water flux in x or y."""
    for i, ii in enumerate(inds):
        if astep[i]:
            qfield[ii] += dirstep[i] / dist[i]
    return qfield


@njit
def _update_absQfield(qfield, dist, inds, astep, Qp_water, dx):
    """Update norm of water flux vector."""
    for i, ii in enumerate(inds):
        if astep[i]:
            qfield[ii] += Qp_water / dx / 2
    return qfield


@njit
def _accumulate_free_surface_walks(free_surf_walk_indices, free_surf_flag, cell_type,
                                   uw, ux, uy, depth, dx, u0, h0, H_SL, S0):
    """Accumulate the free surface by walking parcel paths.

    This routine comprises the hydrodynamic physics-based computations.

    Algorithm is to:
        1. loop through every parcel's directed random walk in series.

        2. for a parcel's walk, unravel the indices and determine whether the
        parcel should contribute to the free surface. Parcels are considered
        contributors if they have reached the ocean and if they are not looped
        pathways.

        3. then, we begin at the downstream end of the parcel's walk and
        iterate up-walk until, determining the `Hnew` for each location.
        Downstream of the shoreline-ocean boundary, the water surface
        elevation is set to the sea level. Upstream of the shoreline-ocean
        boundary, the water surface is determined according to the land-slope
        (:obj:`S0`) and the parcel pathway.

        4. repeat from 2, for each parcel.

    """
    _shape = uw.shape
    Hnew = np.zeros(_shape)
    sfc_visit = np.zeros(_shape)
    sfc_sum = np.zeros(_shape)

    # for every parcel, walk the path of the parcel
    for p, inds in enumerate(free_surf_walk_indices):

        # unravel the indices of the parcel into `xs` and `ys`
        inds_whr = inds[inds > 0]  # where the path has meaningful values
        xs = np.zeros_like(inds_whr)  # x coordinates
        ys = np.zeros_like(inds_whr)  # x coordinates
        for pp, ind_whr in np.ndenumerate(inds_whr):
            xs[pp], ys[pp] = shared_tools.custom_unravel(ind_whr, _shape)

        # determine whether the pathway contributes to the free surface
        Hnew[:] = 0
        if ((cell_type[xs[-1], ys[-1]] == -1) and (free_surf_flag[p] == 1)):
            # if cell is in ocean, H = H_SL (downstream boundary condition)
            Hnew[xs[-1], ys[-1]] = H_SL

            # counting back from last cell visited
            in_ocean = True  # whether we are in the ocean or not
            dH = 0
            for it in range(len(xs) - 2, -1, -1):
                i = xs[it]
                ip = xs[it + 1]
                j = ys[it]
                jp = ys[it + 1]

                # if the parcel has moved at all
                if (i != ip) or (j != jp):
                    # if in the ocean (not reached the shoreline yet)
                    if in_ocean:
                        # see if it is shoreline
                        if ((uw[i, j] > u0 * 0.5) or (depth[i, j] < 0.1 * h0)):
                            in_ocean = False  # passed the shoreline
                    # otherwise, in the delta
                    else:
                        # if no velocity
                        if uw[i, j] == 0:
                            dH = 0  # no change in water surface elevation
                        else:
                            # diff between streamline and parcel path
                            dH = (S0 * (ux[i, j] * (ip - i) * dx +
                                        uy[i, j] * (jp - j) * dx) / uw[i, j])

                # previous cell's surface plus difference in H
                Hnew[i, j] = Hnew[ip, jp] + dH

                # add up # of cell visits
                sfc_visit[i, j] = sfc_visit[i, j] + 1

                # sum of all water surface elevations
                sfc_sum[i, j] = sfc_sum[i, j] + Hnew[i, j]

    return sfc_visit, sfc_sum


@njit
def _smooth_free_surface(Hin, cell_type, Nsmooth, Csmooth):
    """Smooth the free surface.

    Parameters
    ----------
    Hin
        Stage input to the smoothing (i.e., the old stage).

    cell_type
        Type of each cell.

    Nsmooth
        Number of times to smooth the free surface (i.e., stage)

    Csmooth
        Underrelaxation coefficient for smoothing iterations.
    """
    # grab relevant shape information
    L, W = Hin.shape

    # pad the input stage and cell type arrays
    Hin_pad = shared_tools.custom_pad(Hin)
    cell_type_pad = shared_tools.custom_pad(cell_type)

    # create copy of H which is modified in following smoothing
    Htemp = np.copy(Hin)
    for _ in range(Nsmooth):

        # create another copy to refernce as base in Nsmooth iteration
        Hsmth = np.copy(Htemp)
        Hsmth_pad = np.copy(shared_tools.custom_pad(Hsmth))

        # loop through all cells and determine a smoothed index
        for i in range(L):
            for j in range(W):

                # locate non-edge cells
                if cell_type[i, j] != -1:

                    # slice the padded array neighbors
                    cell_type_nbrs = cell_type_pad[
                        i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                    Hsmth_nbrs = Hsmth_pad[
                        i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]

                    # flatten
                    Hsmth_nbrs = np.copy(Hsmth_nbrs.ravel())
                    cell_type_nbrs = np.copy(cell_type_nbrs.ravel())

                    # create a validation array
                    invalid_nbrs = (cell_type_nbrs == -2)  # where land
                    invalid_nbrs[4] = True

                    # invalid cells cannot contribute stage values
                    Hsmth_nbrs[invalid_nbrs] = 0

                    # contributed stage and number of contributing nbrs
                    sumH = np.sum(Hsmth_nbrs)
                    nbcount = np.sum(~invalid_nbrs)

                    # if there are any nbrs
                    if nbcount > 0:
                        # the new stage is the underrelaxed average of nbrs
                        Htemp[i, j] = ((Csmooth * Hsmth[i, j]) +
                                       ((1 - Csmooth) * (sumH / nbcount)))

    # return the smoothed stage
    Hsmth = np.copy(Htemp)
    return Hsmth
