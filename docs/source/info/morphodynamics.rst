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

.. note::
   Incomplete.


.. _sediment-routing-weighting:

Sediment routing weighting
--------------------------

.. note::
   Incomplete.

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

    Total sediment mass is preserved, but individual categories of sand and mud are not. I.e., it is assumed that there is an infinite supply of sand and/or mud to erode and entrain at any location in the model domain.

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
At its core though, model stability depends on the bed elevation rate of change, bot over space and over time. 
Rapid and abrupt bed elevation change trigger numerical instability that can *occasionally* run-away and cause model runs to fail.
A number of processes are included in the DeltaRCM framework to help limit the possibility of failed runs.

.. note::
   Incomplete.


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

.. _reference-volume:

Reference Volume
----------------

The reference volume (:math:`V_0`) impacts model stability. This volume characterizes the volume on one inlet-channel cell, from the channel bed to the water surface:

.. math::

    V_0 = h_0 {\delta_c}^2

where :math:`h_0` is the inlet channel depth (meters) and :math:`\delta_c` is the cell length (meters).


Notes for modeling best practices
=================================

* Stop simulations before the delta reaches the edge of the computational domain. If a distributary channel reaches the domain edge, this channel is likely to become locked in place, and will convey sediment outside the computational domain, thus violating any statements of mass conservation. Generally, simulations that reach the edge of the domain should be discarded. 
* Stop simulations before the delta reaches a grade condition, where the topset slope is equal to background slope `S0`. This is really only an issue for large domains run for long duration.
* Use a sufficient number of water and sediment parcels (> 2000). Too few parcels will result in a rough water surface, irregular sediment deposition, and a rough bed elevation.



References
==========

.. [1] A reduced-complexity model for river delta formation – Part 1: Modeling
       deltas with channel dynamics, M. Liang, V. R. Voller, and C. Paola, Earth
       Surf. Dynam., 3, 67–86, 2015. https://doi.org/10.5194/esurf-3-67-2015
