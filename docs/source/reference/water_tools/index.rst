.. api.water_tools:

*********************************
water_tools
*********************************

.. currentmodule:: pyDeltaRCM.water_tools


The :obj:`~pyDeltaRCM.water_tools.water_tools.route_water` routine manages the water routing.
During :obj:`~pyDeltaRCM.water_tools.water_tools.route_water`, water iteration is repeated a total of :obj:`~pyDeltaRCM.model.DeltaModel.itermax` times.
During each of these iterations of the water routing, the following methods are called *in order*:

.. autosummary::

    water_tools.init_water_iteration
    water_tools.run_water_iteration
    water_tools.compute_free_surface
	water_tools.finalize_water_iteration


Public API methods attached to model
------------------------------------

The following methods are defined in the ``water_tools`` class, of the ``pyDeltaRCM.water_tools`` module.

.. autosummary::

    water_tools

.. autoclass:: water_tools


water_tools helper functions
----------------------------

The following routines are jitted for speed.
They generally take a large number of arrays and constants and return a new array(s) to continue with the model progression within the methods defined above.

.. autofunction:: _get_weight_at_cell_water
.. autofunction:: _choose_next_directions
.. autofunction:: _calculate_new_inds
.. autofunction:: _check_for_loops
.. autofunction:: _update_dirQfield
.. autofunction:: _update_absQfield
.. autofunction:: _accumulate_free_surface_walks
.. autofunction:: _smooth_free_surface
