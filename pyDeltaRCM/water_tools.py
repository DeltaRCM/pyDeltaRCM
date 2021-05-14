
import numpy as np
from numba import njit
import abc

from . import shared_tools

# tools for water routing algorithms


class water_tools(abc.ABC):

    def route_water(self):
        """Water routing main method.

        This is the main method for water routing in the model. It is
        called once per `update()` call.

        Internally, this method calls:
            * :obj:`init_water_iteration`
            * :obj:`run_water_iteration`
            * :obj:`compute_free_surface`
            * :obj:`finalize_water_iteration`
        """
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

    def init_water_iteration(self):
        """Init the water iteration routine.
        """
        _msg = 'Initializing water iteration'
        self.log_info(_msg, verbosity=2)

        self.qxn[:] = 0
        self.qyn[:] = 0
        self.qwn[:] = 0

        self.free_surf_flag[:] = 1  # all parcels begin as valid
        self.free_surf_walk_inds[:] = 0
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

        Example
        -------
        On the initial delta surface, see how ten selected parcels are routed through the domain:

        .. plot:: water_tools/run_water_iteration.py
        """
        _msg = 'Beginning stepping of water parcels'
        self.log_info(_msg, verbosity=2)

        # configure the starting indices for each parcel
        inlet_weights = np.ones_like(self.inlet)
        start_indices = shared_tools.get_start_indices(self.inlet,
                                                       inlet_weights,
                                                       self._Np_water)

        # init parcel step number counter
        _step = 0

        # flux from ghost node
        self.qxn.flat[start_indices] += 1
        self.qyn.flat[start_indices] += 0  # this could be omitted...
        self.qwn.flat[start_indices] += self.Qp_water / self._dx / 2

        # load the initial indices into the walk indices
        self.free_surf_walk_inds[:, _step] = start_indices
        current_inds = np.copy(start_indices)

        # get weights, and flatten for faster access
        self.get_water_weight_array()
        water_weights_flat = self.water_weights.reshape(-1, 9)

        # while any parcles still need to take any steps
        while (sum(current_inds) > 0) & (_step < self.stepmax):

            # step counter increment
            _step += 1

            # check whether storage size needs to be increased
            self.check_size_of_indices_matrix(_step)

            # use water weights and random pick to determine d8 direction
            new_direction = _choose_next_directions(
                current_inds, water_weights_flat)

            # use the new directions for each parcel to determine the new ind
            #   for each parcel
            new_inds = _calculate_new_inds(
                current_inds,
                new_direction,
                self.ravel_walk_flat)

            # determine the step of each parcel made based on the new direction
            dist, istep, jstep, astep = shared_tools.get_steps(
                new_direction,
                self.iwalk_flat,
                self.jwalk_flat)

            # update the discharge field with walk of parcels
            #   this updates flux out of the old inds, and into the new inds
            self.update_Q(
                dist, current_inds, new_inds, astep, jstep, istep,
                update_current=True, update_next=True)

            # check for any loops in the walks of parcels
            #   A loop disqualifies a parcel from the free surface
            #   calculation, but does not stop the parcel from routing, and
            #   thus influencing the qw fields. If a parcel is looped, it
            #   will be given a new location in the domain, along the
            #   mean-transport vector. See function for full description.
            new_inds, looped = _check_for_loops(
                self.free_surf_walk_inds, new_inds, _step, self.L0,
                self.CTR, self.stage - self.H_SL)
            looped = looped.astype(bool)

            # set the current_inds to be the new_inds values
            #   (i.e., take the step)
            current_inds[:] = new_inds[:]

            # invalidate the looped parcels from the free surface
            self.free_surf_flag[looped] = 0

            # check for parcels that have reached the boundary
            boundary = self.check_for_boundary(current_inds)

            # parcels that have looped and reached the boundary are set zero
            #   before saving their indices
            boundary_looped = np.logical_and(looped, boundary)
            current_inds[boundary_looped] = 0

            # Record the parcel pathways for computing the free surface later
            self.free_surf_walk_inds[:, _step] = current_inds

            # parcels that have reached the boundary are set ``ind==0``,
            #   effectively ending the routing of these parcels.
            current_inds[boundary] = 0

            # update the q*n fields for the final step
            #   to ensure flux balanced at domain edge
            if np.any(boundary):
                self.update_Q(
                    dist[boundary], current_inds[boundary],
                    new_inds[boundary], astep[boundary],
                    jstep[boundary], istep[boundary],
                    update_current=False, update_next=True)

    def compute_free_surface(self):
        """Calculate free surface after routing all water parcels.

        This method uses the `free_surf_walk_inds` matrix accumulated
        during the routing of the water parcels (in
        :obj:`run_water_iteration`) to determine the free surface. The
        operations of the free surface computation are placed in a jitted
        function :obj:`_accumulate_free_surface_walks`. Following this
        computation, the free surface is smoothed by steps in
        :obj:`finalize_free_surface`.

        See [1]_ and [2]_ for a complete description of hydrodynamic
        assumptions in the DeltaRCM model.

        Examples
        --------

        The sequence of `compute_free_surface` is depicted in the figures
        below. The first image depicts the "input" to `compute_free_surface`,
        which is the current bed elevation, and the path of each water parcel
        (cyan lines in right image).

        .. plot:: water_tools/compute_free_surface_inputs.py

        The `compute_free_surface` method then calls the
        :obj:`_accumulate_free_surface_walks` function to determine 1) the
        number of times each cell has been visited by a water parcel
        (``sfc_visit``), and 2) the *total sum of expected elevations* of the
        water surface at each cell (``sfc_sum``).

        The output from :obj:`_accumulate_free_surface_walks` is then used to
        calculate a new stage surface (``H_new``) based only on the water
        parcel paths and expected water surface elevations, approximately as
        ``Hnew = sfc_sum / sfc_visit`` ("computed Hnew" in figure below)

        Following this step, a correction is applied forcing the new free
        surface to be below sea level and above or equal to the land surface
        elevation over the model domain ("stage" in figure below). This
        surface `Hnew` is used in the following operation
        :obj:`finalize_free_surface`.

        Finally, the updated water surface is
        combined with the previous timestep's water surface and an
        underrelaxation coefficient
        :obj:`~pyDeltaRCM.model.DeltaModel.omega_sfc`.

        .. plot:: water_tools/compute_free_surface_outputs.py

        .. [1] A reduced-complexity model for river delta formation – Part 1:
               Modeling deltas with channel dynamics, M. Liang, V. R. Voller,
               and C. Paola, Earth Surf. Dynam., 3, 67–86, 2015.
               https://doi.org/10.5194/esurf-3-67-2015

        .. [2] A reduced-complexity model for river delta formation – Part 2:
               Assessment of the flow routing scheme, M. Liang, N. Geleynse,
               D. A. Edmonds, and P. Passalacqua, Earth Surf. Dynam., 3,
               87–104, 2015. https://doi.org/10.5194/esurf-3-87-2015

        """
        _msg = 'Computing free surface from water parcels'
        self.log_info(_msg, verbosity=2)

        self.sfc_visit, self.sfc_sum = _accumulate_free_surface_walks(
            self.free_surf_walk_inds, self.free_surf_flag, self.cell_type,
            self.uw, self.ux, self.uy, self.depth,
            self._dx, self._u0, self.h0, self._H_SL, self._S0)

        # begin from the previous stage
        Hnew = self.eta + self.depth

        # find average water surface elevation for a cell from accumulation
        Hnew[self.sfc_visit > 0] = (self.sfc_sum[self.sfc_visit > 0] /
                                    self.sfc_visit[self.sfc_visit > 0])

        # water surface height not under sea level
        Hnew[Hnew < self._H_SL] = self._H_SL

        # water surface height not below bed elevation
        Hnew[Hnew < self.eta] = self.eta[Hnew < self.eta]

        # set to model field
        self.Hnew = Hnew

        # finalize the free surface (combine and smooth)
        self.finalize_free_surface()

    def finalize_free_surface(self):
        """Finalize the water free surface.

        This method occurs after the initial computation of the free surface,
        and creates the new free surface by smoothing the newly computed free
        surface with a jitted function (:obj:`_smooth_free_surface`), and then
        combining the old surface and new smoothed surface with
        underrelaxation. Finally, a :obj:`flooding_correction` is applied.
        """
        _msg = 'Smoothing and finalizing free surface'
        self.log_info(_msg, verbosity=2)

        # smooth newly calculated free surface
        Hsmth = _smooth_free_surface(
            self.Hnew, self.cell_type, self._Nsmooth, self._Csmooth)

        # combine new smoothed and previous free surf with underrelaxation
        if self._time_iter > 0:
            self.stage = (((1 - self._omega_sfc) * self.stage) +
                          (self._omega_sfc * Hsmth))

        # apply a flooding correction
        self.flooding_correction()

    def finalize_water_iteration(self, iteration):
        """Finish updating flow fields.

        Clean up at end of water iteration
        """
        _msg = 'Finalizing stepping of water parcels'
        self.log_info(_msg, verbosity=2)

        # apply boundary for water surface
        #     Note: not in Matlab implementation
        self.stage[:] = np.maximum(self.stage, self._H_SL)

        # apply an update on the depth to match stage and eta values
        self.depth[:] = np.maximum(self.stage - self.eta, 0)  # never negative

        # update fields for sediment routing
        self.update_flow_field(iteration)
        self.update_velocity_field()

    def check_size_of_indices_matrix(self, it):
        """Check if step path matrix needs to be made larger.

        Initial size of self.free_surf_walk_inds is half of self.stepmax
        because the number of iterations doesn't go beyond
        that for many timesteps.

        Once it reaches it > self.stepmax/2 once, make the size
        self.iter for all further timesteps
        """
        if it >= self.free_surf_walk_inds.shape[1]:
            _msg = 'Increasing size of self.free_surf_walk_inds'
            self.log_info(_msg, verbosity=2)

            indices_blank = np.zeros(
                (np.int(self._Np_water), np.int(self.stepmax / 4)), dtype=int)

            self.free_surf_walk_inds = np.hstack((self.free_surf_walk_inds, indices_blank))

    def get_water_weight_array(self):
        """Get step direction weights for each cell.

        This method is called once, before parcels are stepped, because the
        weights do not change during the stepping of parcels.

        See :doc:`/info/hydrodynamics` for a description of the model design
        and equations/algorithms for how water weights are determined.

        .. note::

            No computation actually occurs in this method. Internally, the
            method calls the jitted :obj:`_get_water_weight_array`, which in
            turn loops through all of the cells in the model domain and
            determines the water weight for that cell with
            :obj:`get_weight_sfc_int` and :obj:`_get_weight_at_cell_water`.

        Examples
        --------

        The following figure shows several examples of locations within the
        model domain, and the corresponding water routing weights determined
        for that location.

        .. plot:: water_tools/water_weights_examples.py
        """
        _msg = 'Computing water weight array'
        self.log_info(_msg, verbosity=2)

        # compiling the water weight array is handled inside a jitted
        #     function below. ~4x faster.
        self.water_weights = _get_water_weight_array(
            self.depth, self.stage, self.cell_type, self.qx, self.qy,
            self.ivec_flat, self.jvec_flat, self.distances_flat,
            self.dry_depth, self.gamma, self._theta_water)

    def update_Q(self, dist, current_inds, next_inds, astep, jstep, istep,
                 update_current=False, update_next=False):
        """Update discharge field values.

        Method is called after one set of water parcel steps.
        """
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

        # inds[self.free_surf_flag == 2] = 0
        boundary = (self.cell_type.flat[inds] == -1)
        return boundary

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
    """Compute water weights for a given cell.

    This is a jitted function called by :func:`_get_water_weight_array`.
    """
    # create a fixed set of bools
    dry = (depth_nbrs < dry_depth)
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

    # sanity check
    if np.any(np.isnan(weight_sfc)) or np.any(np.isnan(weight_int)):
        raise RuntimeError('NaN encountered in input to water weighting. '
                           'Please report error.')

    # initial rebalance weights
    if np.sum(weight_sfc) > 0:
        weight_sfc = weight_sfc / np.sum(weight_sfc)
    if np.sum(weight_int) > 0:
        weight_int = weight_int / np.sum(weight_int)

    weight = gamma * weight_sfc + (1 - gamma) * weight_int
    weight = (depth_nbrs ** theta) * weight

    # enforce disallowed choice to not move
    weight[ctr] = 0

    # correct the weights for random choice
    if np.sum(weight) > 0:
        # if any cells have positive weights, rebalance to sum() == 1
        weight = weight / np.sum(weight)

    elif np.sum(weight) == 0:
        # if all weights are zeros/nan
        if wall[4] == True:  # noqa: E712
            # if current cell is wall cell
            #    just set to zero and pass
            weight[:] = 0
        else:
            # convert to random walk into...
            wetvalid = np.logical_and(~dry, ~invalid)
            nwetvalid = np.sum(wetvalid)
            if nwetvalid > 0:
                # any wet (and valid)
                weight[:] = 0
                weight[wetvalid] = (1 / nwetvalid)
            else:
                # no wet (and valid) neighbors. How did we get here??
                #    Maybe by flooding corrections? Solution is to
                #    allow to walk into any non-land cell, true random
                nnotwall = np.sum(~wall)
                weight[:] = 0
                if nnotwall > 0:
                    weight[~wall] = (1 / nnotwall)
                else:
                    raise RuntimeError('No non-wall cells surrounding cell. '
                                       'Please report error.')

            weight = weight / np.sum(weight)

    else:
        raise RuntimeError('Water sum(weight) less than 0. '
                           'Please report error.')

    # final sanity check
    if np.any(np.isnan(weight)):
        raise RuntimeError('NaN encountered in return from water weighting. '
                           'Please report error.')

    return weight


# @njit('(float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:],'
#       'float32[:], float32[:], float32[:],'
#       'float64, float64, float64)')
@njit
def _get_water_weight_array(depth, stage, cell_type, qx, qy,
                            ivec_flat, jvec_flat, distances_flat,
                            dry_depth, gamma, theta_water):
    """Worker for :obj:`_get_water_weight_array`.

    This is a jitted function which handles the actual computation of looping
    through locations of the model domain and determining the water
    weighting.

    See :meth:`get_water_weight_array` for more information.

    .. note::

        If you are trying to change water weighting behavior, consider
        reimplementing this method, which calls a custom version of
        :func:`_get_weight_at_cell_water`.
    """
    L, W = depth.shape
    pad_stage = shared_tools.custom_pad(stage)
    pad_depth = shared_tools.custom_pad(depth)
    pad_cell_type = shared_tools.custom_pad(cell_type)

    water_weights = np.zeros((L, W, 9))

    for i in range(L):
        for j in range(W):
            stage_nbrs = pad_stage[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
            depth_nbrs = pad_depth[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
            ct_nbrs = pad_cell_type[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]

            weight_sfc, weight_int = shared_tools.get_weight_sfc_int(
                stage[i, j], stage_nbrs.ravel(),
                qx[i, j], qy[i, j], ivec_flat, jvec_flat,
                distances_flat)

            water_weights[i, j] = _get_weight_at_cell_water(
                (i, j), weight_sfc, weight_int,
                depth_nbrs.ravel(), ct_nbrs.ravel(),
                dry_depth, gamma, theta_water)

    return water_weights


@njit('int64[:](int64[:], float64[:,:])')
def _choose_next_directions(inds, water_weights):
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
    next_direction : :obj:`ndarray`
        The direction to move towards the new cell for water parcels, relative
        to the current location. I.e., this is the D8 direction the parcel is
        going to travel in the next stage,
        :obj:`pyDeltaRCM.shared_tools._calculate_new_ind`.
    """
    next_direction = np.zeros_like(inds)
    for p in range(inds.shape[0]):
        ind = inds[p]
        if ind != 0:
            weight = water_weights[ind, :]
            next_direction[p] = shared_tools.random_pick(weight)
        else:
            next_direction[p] = 4

    return next_direction


@njit('int64[:](int64[:], int64[:], int64[:])')
def _calculate_new_inds(current_inds, new_direction, ravel_walk):
    """Calculate the new location (current_inds) of parcels.

    Use the information of the current parcel (`current_inds`) in conjunction
    with the D8 direction the parcel needs to travel (`new_direction`) to
    determine the new current_inds of each parcel.

    In implementation, we use the flattened `ravel_walk` array, but the result
    is identical to unraveling the index, adding `iwalk` and `jwalk` to the
    location and then raveling the index back.

    .. code::

        ind_tuple = shared_tools.custom_unravel(ind, domain_shape)
        new_ind = (ind_tuple[0] + jwalk[newd],
                   ind_tuple[1] + iwalk[newd])
        new_inds[p] = shared_tools.custom_ravel(new_ind, domain_shape)


    """
    # preallocate return array
    new_inds = np.zeros_like(current_inds)

    # loop through every parcel
    for p in range(current_inds.shape[0]):

        # extract current_ind and direction
        ind = current_inds[p]
        newd = new_direction[p]

        # check if the parcel moves
        if newd != 4:
            # if moves, compute new ind for parcel
            new_inds[p] = ind + ravel_walk[newd]
        else:
            # if not moves, set new ind to 0
            #  (should be only those already at 0)
            new_inds[p] = 0

    return new_inds


@njit
def _check_for_loops(free_surf_walk_inds, new_inds, _step,
                     L0, CTR, stage_above_SL):
    """Check for loops in water parcel pathways.

    Look for looping random walks. I.e., this function checks for where a
    parcel will return on its :obj:`new_inds` to somewhere it has already been
    in :obj:`free_surf_walk_inds`. If the loop is found, the parcel is
    relocated along the mean transport vector of the parcel, which is computed
    as the vector from the cell `(0, CTR)` to the new location in `new_inds`.

    This implementation of loop checking will relocate any parcel that has
    looped, but only disqualifies a parcel `p` from contributing to the free
    surface in :obj:`_accumulate_free_surf_walks` (i.e., `looped[p] == 1`) if
    the stage at the looped location is above the sea level in the domain.

    Parameters
    ----------
    free_surf_walk_inds
        Array recording the walk of parcels. Shape is `(:obj:`Np_water`,
        ...)`, where the second dimension will depend on the step number, but
        records each step of the parcel. Each element in the array records the
        *flat* index into the domain.

    new_inds
        Array recording the new index for each water parcel, if the step is
        taken. Shape is `(Np_water, 1)`, with each element recording
        the *flat* index into the domain shape.

    _step
        Step number of water parcels.

    L0
        Domain shape parameter, number of cells inlet length.

    CTR
        Domain shape parameter, index along inlet wall making the center of
        the domain. I.e., `(0, CTR)` is the midpoint across the inlet, along
        the inlet domain edge.

    stage_above_SL
        Water surface elevation minuns the domain sea level.

    Returns
    -------
    new_inds
        An updated array of parcel indicies, where the index of a parcel has
        been changed, if and only if, that parcel was looped.

    looped
        A binary integer array indicating whether a parcel was determined to
        have been looped, and should be disqualified from the free surface
        computation.

    Examples
    --------

    The following shows an example of how water parcels that looped along
    their paths would be relocated. Note than in this example, the parcels are
    artificially forced to loop, just for the sake of demonstration.

    .. plot:: water_tools/_check_for_loops.py

    """
    nparcels = free_surf_walk_inds.shape[0]
    domain_shape = stage_above_SL.shape
    domain_min_x = domain_shape[0] - 2
    domain_min_y = domain_shape[1] - 2
    L0_ind_cut = ((L0) * domain_shape[1])-1

    looped = np.zeros_like(new_inds)

    stage_v_SL = np.abs(stage_above_SL) < 1e-1  # true if they are same

    # if the _step number is larger than the inlet length
    if (_step > L0):
        # loop though every parcel walk
        for p in np.arange(nparcels):
            new_ind = new_inds[p]  # the new index of the parcel
            full_walk = free_surf_walk_inds[p, :]  # the parcel's walk
            nonz_walk = full_walk[full_walk > 0]   # where non-zero
            relv_walk = nonz_walk[nonz_walk > L0_ind_cut]

            if (new_ind > 0):

                # determine if has a repeat ind
                has_repeat_ind = False
                for _, iind in enumerate(relv_walk):
                    if iind == new_ind:
                        has_repeat_ind = True
                        break

                if has_repeat_ind:
                    # handle when a loop is detected
                    px0, py0 = shared_tools.custom_unravel(
                        new_ind, domain_shape)

                    # compute a new location for the parcel along the
                    #   mean-transport vector
                    Fx = px0 - 1
                    Fy = py0 - CTR
                    Fw = np.sqrt(Fx**2 + Fy**2)

                    # relocate the parcel along mean-transport vector
                    if Fw != 0:
                        px = px0 + int(np.round(Fx / Fw * 5.))
                        py = py0 + int(np.round(Fy / Fw * 5.))

                    # limit the new px and py to beyond the inlet, and
                    #   away from domain edges
                    px = np.minimum(domain_min_x, np.maximum(px, L0))
                    py = np.minimum(domain_min_y, np.maximum(1, py))

                    # ravel the index for return
                    nind = shared_tools.custom_ravel((px, py), domain_shape)
                    new_inds[p] = nind

                    # only disqualify the parcel if it has not reached sea
                    #   level by the time it loops
                    if not stage_v_SL[px0, py0]:
                        looped[p] = 1  # this parcel is looped

    return new_inds, looped


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
def _accumulate_free_surface_walks(free_surf_walk_inds, free_surf_flag,
                                   cell_type, uw, ux, uy, depth,
                                   dx, u0, h0, H_SL, S0):
    """Accumulate the free surface by walking parcel paths.

    This routine comprises the hydrodynamic physics-based computations.

    Algorithm is to:
        1. loop through every parcel's directed random walk in series.

        2. for a parcel's walk, unravel the indices and determine whether the
           parcel should contribute to the free surface. Parcels are
           considered contributors if they have reached the ocean and if they
           are not looped pathways.

        3. then, we begin at the downstream end of the parcel's walk and
           iterate up-walk until, determining the `Hnew` for each location.
           Downstream of the shoreline-ocean boundary, the water surface
           elevation is set to the sea level. Upstream of the shoreline-ocean
           boundary, the water surface is determined according to the
           land-slope (:obj:`S0`) and the parcel pathway.

        4. repeat from 2, for each parcel.


    Examples
    --------

    The following shows an example of the walk of a few water parcels, along
    with the resultant computed water surface.

    .. plot:: water_tools/_accumulate_free_surface_walks.py

    """
    _shape = uw.shape
    Hnew = np.zeros(_shape)
    sfc_visit = np.zeros(_shape)
    sfc_sum = np.zeros(_shape)

    # for every parcel, walk the path of the parcel
    for p, inds in enumerate(free_surf_walk_inds):

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

    Examples
    --------
    The following shows an example of the water surface smoothing process.

    .. plot:: water_tools/_smooth_free_surface.py
    """
    # grab relevant shape information
    L, W = Hin.shape

    # pad the input stage and cell type arrays
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
                    Hsmth_nbrs = Hsmth_nbrs.flatten()
                    cell_type_nbrs = cell_type_nbrs.flatten()

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
