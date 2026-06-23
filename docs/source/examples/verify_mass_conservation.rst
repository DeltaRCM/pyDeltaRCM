Verifying sediment mass conservation 
====================================

Here, we learn about the mass conservation principles applied in the model.

The DeltaRCM formulation is explictly not conservative of mass (neither water
nor sediment) [1]_. This is generally not a problem, as the model is internally
consistent, that is, the rules don't change over time or with different
parameter/boundary conditions. 

However, it is fairly common to have issues with sediment mass conservation in
atypical model domain configurations. Additionally, some modelers may aim to
compare with real-world deltas, and therefore aim to keep sediment mass
conservation as strict as reasonably possible.  

.. note::

   We are not aware of any straightforward way to formulate the model for water
   mass conservation. Relaxing the requirement for water mass conservation is 
   part of what makes the DeltaRCM formulation computationally efficient and
   attractive to many modelers.

There are three primary areas of concern when discussion sediment mass
conservation in the DeltaRCM framework. With the default :code:`DeltaModel` and
default model parameters:

1. sediment routing terminates iteration without depositing all stored sediment
   when :code:`stepmax` iterations are reached,
2. sediment parcels that reach the edge of the model domain leave the domain
   without depositing sediment. 
3. sediment is added or removed at the inlet when bed elevation is updated to
   :code:`stage - h_0` on each iteration, and

Notably, the second point is consistent with a mass conservation formulation
that considers the domain the control volume. It is brought up here, however,
because this can be a source of confusion for new modelers, and is why
simulations where the delta reaches the edge of the computational domain should
be discarded. 

Let's examine the magnitude of these "lost", "exported", and "at inlet" sediment
volumes. The below script generates tables tracking the volume of sediment
conservation in the model under these relevant parameters. We track volume
because deposit porosity is neglected in the original DeltaRCM formulation. 


.. literalinclude:: verify_mass_conservation.py
   :language: python

.. program-output:: python examples/verify_mass_conservation.py

Here, it is clear that default model parameters lead to an overall high degree
of sediment mass conservation. Be careful not to have the domain too small, or
large volumes of sediment will be exported. Similarly, setting :code:`stepmax`
to too small of a value can lead to large volume of lost sediment. 

.. hint::
   
   If the volume of sediment changing lost to :code:`stepmax` iterations, or due to
   atypical model domain configurations, consider setting the parameter
   :code:`force_deposit` flag to :code:`True`. This flag forces sediment to deposit
   in place when `stepmax` iterations are reached; it can create new instabilities,
   but improves mass conservation.

.. hint::
   
   If the volume of sediment changing at the inlet is significant for your
   modeling, consider reimplementing :code:`finalize_timestep()` to eliminate or
   modify this boundary condition.

.. [1] A reduced-complexity model for river delta formation --- Part 1: Modeling
       deltas with channel dynamics, M. Liang, V. R. Voller, and C. Paola, Earth
       Surf. Dynam., 3, 67--86, 2015. https://doi.org/10.5194/esurf-3-67-2015

