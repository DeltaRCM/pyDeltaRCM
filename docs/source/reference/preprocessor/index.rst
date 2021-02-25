.. api.preprocessor:

*********************************
Preprocessor
*********************************

The high-level API is principally defined in ``pyDeltaRCM.preprocessor``. 



Preprocessor classes and API
----------------------------

.. currentmodule:: pyDeltaRCM.preprocessor

.. autosummary:: 
    :toctree: ../../_autosummary

    PreprocessorCLI
    Preprocessor
    BasePreprocessor

Internally, jobs configured by the preprocessors are handled by `Job` classes. You likely do not need to interact with these classes, unless you are trying to implement a different execution of the model; for example, trying to run the model until a given deposit thickness is reached, rather than time or number of timesteps.

.. autosummary:: 
    :toctree: ../../_autosummary

    _SerialJob
    _ParallelJob
    _BaseJob

Preprocessor function and utilities
-----------------------------------

.. autofunction:: preprocessor_wrapper
.. autofunction:: write_yaml_config_to_file
.. autofunction:: scale_relative_sea_level_rise_rate
