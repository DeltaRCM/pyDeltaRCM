*************
Hydrodynamics
*************

pyDeltaRCM approximates hydrodynamics through the use of a weighted random walk.
See [1]_ and [2]_ for a complete description of hydrodynamic assumptions in the DeltaRCM model.
In this documentation, we focus on the details of *model implementation*, rather than *model design*.

Routing individual water parcels
================================

.. note::

   Incomplete.


Combining parcels into free surface
===================================

Following the routing of water parcels, these walks must be converted in some meaningful way to a model field representing a free surface (i.e., the water stage).
First, the :obj:`compute_free_surface` is called, which takes as input the current bed elevation, and the path of each water parcel (top row in figure below).

.. plot:: water_tools/compute_free_surface_inputs.py

The :obj:`compute_free_surface` method internally calls the :obj:`_accumulate_free_surface_walks` function to determine 1) the number of times each cell has been visited by a water parcel
(``sfc_visit``), and 2) the *total sum of expected elevations* of the water surface at each cell (``sfc_sum``).
:obj:`_accumulate_free_surface_walks` itself iterates through each water parcel, beginning from the end-point of the path, and working upstream; note that parcels that have been determined to "loop" (:obj:`_check_for_loops` and described above) are excluded from computation in determining the free surface.
While downstream of the land-ocean boundary (determined by a depth-or-velocity threshold), the water surface elevation is assumed to be ``0``, whereas upstream of this boundary, the predicted elevation of the water surface is determined by the distance from the previously identified water surface elevation and the background land slope (:obj:`S0`), such the the water surface maintains an approximately constant slope for each parcel pathway.

.. plot:: water_tools/_accumulate_free_surface_walks.py

The algorithm tracks the number of times each cell has been visited by a water parcel (``sfc_visit``), and the *total sum of expected elevations* of the water surface at each cell (``sfc_sum``), by adding the predicted surface elevation of each parcel step while iterating through each step of each parcel.

Next, the output from :obj:`_accumulate_free_surface_walks` is used to calculate a new stage surface (``H_new``) based only on the water parcel paths and expected water surface elevations, approximately as ``H_new = sfc_sum / sfc_visit``.
The updated water surface is combined with the previous timestep's water surface and an underrelaxation coefficient (:obj:`_omega_sfc`).

.. plot:: water_tools/compute_free_surface_outputs.py

With a new free surface computed, a few final operations prepare the surface for boundary condition updates and eventually being passed to the sediment routing operations.
A non-linear smoothing operation is applied to the free surface, whereby wet cells are iteratively averaged with neighboring wet cells to yield an overall smoother surface.
The smoothing is handled by :obj:`_smooth_free_surface` and depends on the number of iterations (:obj:`Nsmooth`) and a weighting coefficient (:obj:`Csmooth`).

.. plot:: water_tools/_smooth_free_surface.py


.. todo:: add component describing the smoothing.

.. todo:: add component describing the flooding correction.




Finalizing and boundary conditions to sediment routing
======================================================

.. todo::

   Incomplete. Need to describe the updating of depth from stage, limiting everything to above H_SL, updating velocity and discharge fields, etc.


References
==========

.. [1] A reduced-complexity model for river delta formation – Part 1: Modeling
       deltas with channel dynamics, M. Liang, V. R. Voller, and C. Paola, Earth
       Surf. Dynam., 3, 67–86, 2015. https://doi.org/10.5194/esurf-3-67-2015

.. [2] A reduced-complexity model for river delta formation – Part 2:
       Assessment of the flow routing scheme, M. Liang, N. Geleynse,
       D. A. Edmonds, and P. Passalacqua, Earth Surf. Dynam., 3, 87–104, 2015.
       https://doi.org/10.5194/esurf-3-87-2015
