**************
Morphodynamics
**************

pyDeltaRCM approximates sediment dispersal through the use of a weighted random
walk dictated by water flux.
In turn, sediment dispersal drives bed elevation change in the model domain by mass conservation.


==================
Sediment Transport
==================

.. note::
   Incomplete.

===============
Model Stability
===============

Model stability depends on...

.. note::
   Incomplete.

.. _reference_volume:

Reference Volume
----------------

The reference volume (:math:`V_0`) impacts model stability. This volume characterizes the volume on one inlet-channel cell, from the channel bed to the water surface:

.. math::

    V_0 = h_0 {\\delta_c}^2,

where :math:`h_0` is the inlet channel depth (meters) and :math:`\\delta_c` is the cell length (meters).
