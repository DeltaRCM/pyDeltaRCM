Available Model Hooks
=====================

.. important::

    Have a suggestion for another `hook` in the DeltaModel? Get in touch via
    the GitHub issue tracker!


Method hooks
------------

As described extensively in the :ref:`User Guide on how to customize the model
<customize_the_model>`, hooks are methods in the model sequence that do
nothing by default, but can be augmented to provide arbitrary desired
behavior in the model. Hooks have been integrated throughout the model
initialization and update sequences, to allow the users to achieve complex
behavior at various stages of the model sequence.

A complete list of hooks in the model follows:

.. currentmodule:: pyDeltaRCM.hook_tools

.. csv-table:: Available model method hooks
    :header: "Initializing", "Updating"
    :widths: 40, 40

    :obj:`~hook_tools.hook_import_files`, :obj:`~hook_tools.hook_solve_water_and_sediment_timestep`
    :obj:`~hook_tools.hook_process_input_to_model`, :obj:`~hook_tools.hook_apply_subsidence`
    :obj:`~hook_tools.hook_create_other_variables`, :obj:`~hook_tools.hook_finalize_timestep`
    :obj:`~hook_tools.hook_create_domain`, :obj:`~hook_tools.hook_route_water`
    :obj:`~hook_tools.hook_load_checkpoint`, :obj:`~hook_tools.hook_init_water_iteration`
    :obj:`~hook_tools.hook_init_output_file`, :obj:`~hook_tools.hook_run_water_iteration`
    , :obj:`~hook_tools.hook_compute_free_surface`
    , :obj:`~hook_tools.hook_finalize_water_iteration`
    , :obj:`~hook_tools.hook_route_sediment`
    , :obj:`~hook_tools.hook_route_all_sand_parcels`
    , :obj:`~hook_tools.hook_topo_diffusion`
    , :obj:`~hook_tools.hook_route_all_mud_parcels`
    , :obj:`~hook_tools.hook_compute_sand_frac`
    , :obj:`~hook_tools.hook_after_route_water`
    , :obj:`~hook_tools.hook_after_route_sediment`
    , :obj:`~hook_tools.hook_output_data`
    , :obj:`~hook_tools.hook_output_checkpoint`


Array hooks
-----------

The `DeltaModel` also incorporates a few arrays that enable a similar effect
to method hooks. These arrays are initialized with all ``1`` or all ``0``
values by default (depending on the scenario), and provide a convenient
location to augment the model with varied behavior.

A complete list of behavior-modifying arrays in the model follows:

.. csv-table:: Available model array hooks
    :header: "Array name", "function", "default value"
    :widths: 20, 50, 10

    `mod_water_weight`, modifies the neighbor-weighting of water parcels during routing according to ``(depth * mod_water_weight)**theta_water``, 1

