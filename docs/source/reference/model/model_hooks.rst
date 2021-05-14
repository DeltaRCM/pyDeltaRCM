Available Model Hooks
=====================

A complete list of hooks in the model follows:

.. currentmodule:: pyDeltaRCM.hook_tools

.. csv-table:: Available model hooks
    :header: "Initializing", "Updating"
    :widths: 40, 40

    :obj:`~hook_tools.hook_import_files`, :obj:`~hook_tools.hook_solve_water_and_sediment_timestep`
    :obj:`~hook_tools.hook_process_input_to_model`, :obj:`~hook_tools.hook_apply_subsidence`
    :obj:`~hook_tools.hook_create_other_variables`, :obj:`~hook_tools.hook_finalize_timestep`
    :obj:`~hook_tools.hook_create_domain`, :obj:`~hook_tools.hook_route_water`
    :obj:`~hook_tools.hook_load_checkpoint`, :obj:`~hook_tools.hook_init_water_iteration`
    :obj:`~hook_tools.hook_output_data`, :obj:`~hook_tools.hook_run_water_iteration`
    :obj:`~hook_tools.hook_output_checkpoint`, :obj:`~hook_tools.hook_compute_free_surface`
    :obj:`~hook_tools.hook_init_output_file`, :obj:`~hook_tools.hook_finalize_water_iteration`
    , :obj:`~hook_tools.hook_route_sediment`
    , :obj:`~hook_tools.hook_route_all_sand_parcels`
    , :obj:`~hook_tools.hook_topo_diffusion`
    , :obj:`~hook_tools.hook_route_all_mud_parcels`
    , :obj:`~hook_tools.hook_compute_sand_frac`
