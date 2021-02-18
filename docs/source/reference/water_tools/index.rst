.. api.water_tools:

*********************************
water_tools
*********************************

The tools are defined in ``pyDeltaRCM.water_tools``. 


Public API methods attached to model
------------------------------------

.. currentmodule:: pyDeltaRCM.water_tools

.. autosummary:: 
    :toctree: ../../_autosummary

    water_tools


water_tools helper functions
----------------------------

Note that these routines are jitted for speed. They generally take a large
number of arrays and constants and return a new array(s) to continue with the
model progression.

.. autofunction:: _get_weight_at_cell_water
.. autofunction:: _choose_next_directions
.. autofunction:: _calculate_new_inds
.. autofunction:: _check_for_loops
.. autofunction:: _update_dirQfield
.. autofunction:: _update_absQfield
.. autofunction:: _accumulate_free_surface_walks
.. autofunction:: _smooth_free_surface
