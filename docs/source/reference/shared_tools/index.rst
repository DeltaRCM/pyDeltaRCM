.. api.shared_tools:

*********************************
shared_tools
*********************************

.. currentmodule:: pyDeltaRCM.shared_tools


.. todo:: add paragraph description of the module


The tools are defined in ``pyDeltaRCM.shared_tools``.


Shared functions
----------------

This module defines several functions that are used throughout the model, and so are organized here for convenience.

.. autofunction:: get_random_uniform
.. autofunction:: get_start_indices
.. autofunction:: get_steps
.. autofunction:: random_pick
.. autofunction:: custom_unravel
.. autofunction:: custom_ravel
.. autofunction:: custom_pad
.. autofunction:: get_weight_sfc_int
.. autofunction:: custom_yaml_loader


Time scaling functions
----------------------

Scaling of real-world time and model time is an important topic covered in detail in :doc:`/info/modeltime`.
Several functions are defined here which can help with scaling between model and real-world time.

.. autofunction:: scale_model_time
.. autofunction:: _scale_factor


Utilities
---------

Additionally, functions defined in ``pyDeltaRCM.shared_tools`` manage the random state of the model, and help with documentation and version management.

.. autofunction:: set_random_seed
.. autofunction:: get_random_state
.. autofunction:: set_random_state
.. autofunction:: _docs_temp_directory
.. autofunction:: _get_version
