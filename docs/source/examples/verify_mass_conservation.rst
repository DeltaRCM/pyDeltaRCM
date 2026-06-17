
Here, we check the model for sediment mass conservation. The DeltaRCM
formulation is not conservative of mass by default; that is, sediment parcels
may terminate iteration without depositing all stored sediment. For example,
sediment is "lost" when the maximum number of allowable iterations
:code:`stepmax` is reached. Additionally, sediment is removed at the inlet
boundary condition, when bed elevation is updated to :code:`stage - h_0` on each
iteration. Lastly, sediment parcels that reach the edge of the model domain are
"exported" and removed from the model; this *is* mass conservative, but may lead
to unexpected results if the model domain is imrpoperly configured.  

The below script generates a table to demonstrate sediment mass conservation in
the model under these relevant parameters.

.. literalinclude:: verify_mass_conservation.py
   :language: python

.. program-output:: python verify_mass_conservation.py
