.. api.debug_tools:

***********
debug_tools
***********

.. currentmodule:: pyDeltaRCM.debug_tools

The debugging tools are defined in ``pyDeltaRCM.debug_tools``. 

.. todo::

    Add paragraph description of the module. What stages are defined here generally?


Public API methods attached to model
------------------------------------

The following methods are defined in the ``debug_tools`` class, of the ``pyDeltaRCM.debug_tools`` module.
They are then attached as methods of the `DeltaModel` and can be called at any time during run.

.. autosummary::

    debug_tools

.. autoclass:: debug_tools


Public plotting methods
-----------------------

The functions defined below are (generally) the actual workers that handle the plotting for the methods defined above and attached to model.
We expose these functions here because they are be useful in the documentation, where they are used extensively.

.. autofunction:: plot_domain
.. autofunction:: plot_ind
.. autofunction:: plot_line
