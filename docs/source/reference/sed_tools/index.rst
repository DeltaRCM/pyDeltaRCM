.. api.sed_tools:

*********************************
sed_tools
*********************************

.. currentmodule:: pyDeltaRCM.sed_tools

.. todo:: add paragraph description of the module


The tools are defined in ``pyDeltaRCM.sed_tools``. 


Public API methods attached to model
------------------------------------

The following methods are defined in the ``sed_tools`` class, of the ``pyDeltaRCM.sed_tools`` module. 

.. autosummary::

    sed_tools

.. autoclass:: sed_tools


Router classes
--------------

The following classes are defined in the ``pyDeltaRCM.sed_tools`` module. These sediment routing classes are jitted for speed.

.. autosummary:: 
    :toctree: ../../_autosummary

    SandRouter
        :members:
        :inherited-members:
        :private-members:
    
    MudRouter
        :members:
        :inherited-members:
        :private-members:
    
    BaseRouter
        :members:
        :inherited-members:
        :private-members:

sed_tools helper functions
----------------------------

Additionally, the sediment parcel step-weighting function is defined at the module level in ``pyDeltaRCM.sed_tools``.

.. autofunction:: _get_weight_at_cell_sediment
