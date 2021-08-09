**************
Morphodynamics
**************

.. currentmodule:: pyDeltaRCM.sed_tools

pyDeltaRCM approximates sediment dispersal through the use of a weighted random
walk dictated by water flux.
In turn, sediment dispersal drives bed elevation change in the model domain by mass conservation.

See [1]_ for a complete description of morphodynamic assumptions in the DeltaRCM model.
In this documentation, we focus on the details of *model implementation*, rather than *model design*.


.. _sediment-transport:

==================
Sediment Transport
==================

Sediment transport in the model is computed according to an excess stress approach.
Conceptually, sand is routed as bed-material load, and mud is routed as fully suspended load.

For sand parcels, the *transport capacity* is determined by the scaling between sediment flux and flow velocity, and takes the form of the Meyer-Peter and Müller (1948) [3]_ formula:

.. math::

      q_{s\_cap} = q_{s0} \frac{u^\beta_{loc}}{u^\beta_0},

where :math:`u_{loc}` is the depth averaged flow velocity in the cell, :math:`beta` is an exponent set to 3 by default (:obj:`~pyDeltaRCM.model.DeltaModel.beta`), and :math:`q_{s0}` is the unit-width upstream sand flux input at the inlet channel.
At each step of the model domain, sand is either eroded or deposited to the bed depending on the local flow velocity :math:`u_{loc}` and local sediment transport :math:`q_{s\_loc}`. 
Sand is deposited where local transport (i.e., the sand put into that cell from upstream) is greater than the cell transport capacity :math:`q_{s\_loc} > q_{s\_cap}`.
Sand is eroded from the bed when the local velocity is greater than the threshold erosion velocity (:obj:`~pyDeltaRCM.model.DeltaModel.coeff_U_ero_sand`) **and** the local transport is less than the local transport capacity.


Mud parcels do not have any local capacity (i.e., fully suspended washload transport). 
At each parcel step, mud is either eroded or deposited (or neither), depending on the relative value of local flow velocity :math:`u_{loc}` and the threshold erosion and deposition values (:obj:`~pyDeltaRCM.model.DeltaModel.coeff_U_ero_mud` and :obj:`~pyDeltaRCM.model.DeltaModel.coeff_U_dep_mud`).

.. note:: 

      A complete conceptual description of sediment erosion and deposition routing rules can be found in the original DeltaRCM reference Liang et al., 2015 [1]_.


.. _sediment-routing-weighting:

Sediment routing weighting
--------------------------

Sediment routing probability for a given cell :math:`j` to neighbor cell :math:`i` is computed according to:

.. math::

    w_i = \frac{\frac{1}{R_i} \max(0, \mathbf{F}\cdot\mathbf{d_i})}{\Delta i},

where :math:`\mathbf{F}` is the local routing direction and :math:`\mathbf{d_i}` is a unit vector pointing to neighbor :math:`i` from cell :math:`j`, and :math:`\Delta_i` is the cellular distance to neighbor :math:`i` (:math:`1` for cells in main compass directions and :math:`\sqrt{2}` for corner cells.
:math:`R_i` is a resistance estimated as an inverse function of local water depth (:math:`h_i`):

.. math::

    R_i = \frac{1}{{h_i}^\theta}.

Here, :math:`\theta` takes the value of :obj:`~pyDeltaRCM.model.DeltaModel.coeff_theta_sand` for sand routing probabilities, and :obj:`~pyDeltaRCM.model.DeltaModel.coeff_theta_mud` for mud routing.


.. plot:: sed_tools/sediment_weights_examples.py


============================
Changes in the bed elevation
============================

Along the walk of a sediment parcel, the sediment parcel volume is modulated on each step, according to the sediment transport rules described above in :ref:`sediment-transport`.
As the volume of the sediment parcel changes, the channel bed elevation at the current parcel location is updated to reflect this volume change (:obj:`~sed_tools.BaseRouter._update_fields`), i.e., the bed is eroded or sediment is deposited on the bed.
The vertical change in the channel bed is dictated by sediment mass conservation (i.e., Exner equation) and is equal to:

.. math::

    \Delta \eta = \Delta V / dx^2

where :math:`\Delta V` is the volume of sediment to be eroded or deposited from the bed at a given cell along the parcel walk.

.. note::

    Total sediment mass is preserved during erosion, but individual categories of sand and mud are not. I.e., it is assumed that there is an infinite supply of sand and/or mud to erode and entrain at any location in the model domain.

Following a change in the bed elevation, the local flow depth is updated and then local flow velocity is updated according to fluid mass conservation (i.e., ``uw = qw / h``; :obj:`~sed_tools.BaseRouter._update_fields`; [1]_).

Sediment parcels are routed through the model domain step-by-step and in serial, such that changes in the bed elevation caused by one sediment parcel will affect the weighted random walks of all subsequent sediment parcels (:ref:`sediment-routing-weighting`), due to the updated flow field.

Sediment parcel routing is handled by first routing all sand parcels, applying a topographic diffusion (see below and :meth:`~sed_tools.topo_diffusion`), and then routing all mud parcels.
The impact of routing *all* sand and mud parcels on bed elevation is shown in the table below.

.. _sand-mud-route-comparison:

.. table::

    +-------------------------------------------+-----------------------------------------------+----------------------------------------------+
    | initial bed                               | :meth:`~sed_tools.route_all_sand_parcels`     | :meth:`~sed_tools.route_all_mud_parcels`     |
    +===========================================+===============================================+==============================================+
    | .. plot:: sed_tools/_initial_bed_state.py | .. plot:: sed_tools/route_all_sand_parcels.py | .. plot:: sed_tools/route_all_mud_parcels.py |
    +-------------------------------------------+-----------------------------------------------+----------------------------------------------+

.. _model-stability:

===============
Model Stability
===============

Model stability depends on a number of conditions.
At its core though, model stability depends on the bed elevation rate of change over space and time. 
Rapid and/or abrupt bed elevation change trigger numerical instability that can *occasionally* run-away and cause model runs to fail.
A number of processes are included in the DeltaRCM framework to help limit the possibility of failed runs.


.. _reference-volume:

Reference Volume
----------------

The reference volume if the foundational unit (:math:`V_0`) impacting model stability.
This value characterizes the volume on one inlet-channel cell, from the channel bed to the water surface:

.. math::

    V_0 = h_0 {\delta_c}^2

where :math:`h_0` is the inlet channel depth (meters) and :math:`\delta_c` is the cell length (meters).


Time stepping
-------------

Perhaps most important to model stability is the model timestep. 
Recall that for each iteration, the number of parcels and input sediment discharge (as `h0 u0 (c0_percent)/100`) are set by the user (fixed).
Therefore, to control stability, the duration of time represented by each iteration (i.e., the timestep) is determined such that changes in bed elevation per iteration are small.
The model timestep is determined as:

.. math::

    dt = dV_s / N_{p,sed}

where :math:`N_{p,sed}` is the number of sediment parcels, :math:`dV_s` is a characteristic sediment volume, based on the reference volume and inlet width as :math:`dV_s = 0.1 N_0^2 V_0`, where :math:`N_0` is the number of cells across the inlet.


Limiting bed elevation change
-----------------------------

At each sediment parcel step, bed elevation change is limited to 1/4 of the local flow depth. 
Additionally, an edge case where repeated channel bed deposition creates a local `depth` < 0 is restricted by enforcing zero deposition if the `depth` < 0.
These regulations are implemented in the `BaseRouter` class, as :obj:`~pyDeltaRCM.sed_tools.BaseRouter._limit_Vp_change`.


.. topographic-diffusion:

Topographic diffusion
---------------------

Abrupt change in bed elevation (i.e., steep local bed slope) may lead to numerical instability. 
To prevent this, a topographic diffusion is applied immediately following the routing of all sand parcels in the model sequence.

.. hint::

    Topographic diffusion is applied between routing sand parcels and routing mud parcels.

In implementation, topographic smoothing convolves topography with `3x3` cell kernels configured to a diffusive behavior.
The diffusion is repeated over the entire model domain :obj:`~pyDeltaRCM.DeltaModel.N_crossdiff` times.
In the following example, :obj:`~pyDeltaRCM.DeltaModel.N_crossdiff` takes the :doc:`default value </reference/model/yaml_defaults>`.

.. plot:: sed_tools/topo_diffusion.py

The impact of topographic diffusion is minor compared to the bed elevation change driven by parcel erosion or deposition (:ref:`sand and mud routing effects <sand-mud-route-comparison>`).


Notes for modeling best practices
=================================

* Stop simulations before the delta reaches the edge of the computational domain. Delta channel dynamics are changed when a distributary channel reaches the domain edge, because sediment is conveyed to outside the computational domain where it no longer feeds back on channel development. Channels become "locked" in place in this scenario [2]_, because the domain edge is an infinite sediment sink, and therefore rendering invalid any assumptions about stationarity of delta dynamics and/or stratigraphy. Moreover, the downstream water surface boundary condition (`H_SL`) will be violated if a channel reaches the domain edge. Generally, simulations that reach the edge of the domain should be discarded.
* Stop simulations before the delta reaches a condition where the topset slope is equal to the background slope parameter (`S0`). When the background slope is reached, the transport capacity of sediment through the delta is diminished such that channels "clog" up and trigger model instabilities. This is really only an issue for large domains run for long duration.
* Use a sufficient number of water and sediment parcels (> 2000). Too few parcels will result in a rough water surface, irregular sediment deposition, and a rough bed elevation.



References
==========

.. [1] A reduced-complexity model for river delta formation – Part 1: Modeling
       deltas with channel dynamics, M. Liang, V. R. Voller, and C. Paola, Earth
       Surf. Dynam., 3, 67–86, 2015. https://doi.org/10.5194/esurf-3-67-2015

.. [2] Liang, M., Kim, W., and Passalacqua, P. (2016), How much subsidence is
       enough to change the morphology of river deltas?, Geophysical Research Letters, 43, 10,266--10,276, doi:10.1002/2016GL070519.

.. [3] Meyer-Peter, E. and Müller, R.: Formulas for bed-load transport, in: 
       Proceedings of the 2nd Meeting of IAHSR, Stockholm, Sweden, 39–64, 1948.