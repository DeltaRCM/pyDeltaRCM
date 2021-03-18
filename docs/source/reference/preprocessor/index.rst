.. api.preprocessor:

*********************************
Preprocessor
*********************************

.. currentmodule:: pyDeltaRCM.preprocessor

The high-level API is principally defined in ``pyDeltaRCM.preprocessor``. 

.. todo::

    add paragraph description of the module. What can we do with this? Why does it exist? Link to the relevant documentation in the user guide and examples.


Preprocessor classes and API
----------------------------

The following classes are defined in the ``pyDeltaRCM.preprocessor`` module and enable the high-level model API to work at both the command line and as a Python object.

.. autosummary:: 
    :toctree: ../../_autosummary

    PreprocessorCLI
        :members:
        :inherited-members:

    Preprocessor
        :members:
        :inherited-members:
    
    BasePreprocessor

Preprocessor function and utilities
-----------------------------------

.. todo:: add description, what are these?

.. autofunction:: preprocessor_wrapper
.. autofunction:: scale_relative_sea_level_rise_rate
.. autofunction:: _write_yaml_config_to_file
