.. api.init_tools:

**********
init_tools
**********

.. currentmodule:: pyDeltaRCM.init_tools


The model initialization is managed by :obj:`~pyDeltaRCM.model.DeltaModel.__init__`, but the actual initialization is mostly handled by routines in `init_tools`.
The major steps of initialization are:

.. autosummary::

    init_tools.import_files
    init_tools.init_logger
    init_tools.process_input_to_model
    init_tools.determine_random_seed
    init_tools.create_other_variables
    init_tools.create_domain
    init_tools.init_sediment_routers
    init_tools.init_subsidence

and then depending on the checkpointing configuration, the following methods may be called:

.. autosummary::

    init_tools.load_checkpoint
    init_tools.init_output_file
    pyDeltaRCM.iteration_tools.iteration_tools.output_data
    pyDeltaRCM.iteration_tools.iteration_tools.output_checkpoint
    pyDeltaRCM.iteration_tools.iteration_tools.log_model_time


Public API methods attached to model
------------------------------------

The following methods are defined in the ``init_tools`` class, of the ``pyDeltaRCM.init_tools`` module. 

.. currentmodule:: pyDeltaRCM.init_tools

.. autosummary::

    init_tools

.. autoclass:: init_tools
