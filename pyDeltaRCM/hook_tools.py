import abc


class hook_tools(abc.ABC):
    """Tools defining various hooks in the model.

    For an explanation on model hooks, see the :doc:`User Guide
    </guides/user_guide>`.
    """

    def hook_import_files(self):
        """Hook :obj:`~pyDeltaRCM.init_tools.init_tools.import_files`.

        Called immediately before
        :obj:`~pyDeltaRCM.init_tools.init_tools.import_files`.
        """
        pass

    def hook_process_input_to_model(self):
        """Hook :obj:`~pyDeltaRCM.init_tools.init_tools.process_input_to_model`.

        Called immediately before
        :obj:`~pyDeltaRCM.init_tools.init_tools.process_input_to_model`.
        """
        pass

    def hook_create_other_variables(self):
        """Hook :obj:`~pyDeltaRCM.init_tools.init_tools.create_other_variables`.

        Called immediately before
        :obj:`~pyDeltaRCM.init_tools.init_tools.create_other_variables`.
        """
        pass

    def hook_create_domain(self):
        """Hook :obj:`~pyDeltaRCM.init_tools.init_tools.create_domain`.

        Called immediately before
        :obj:`~pyDeltaRCM.init_tools.init_tools.create_domain`.
        """
        pass

    def hook_load_checkpoint(self):
        """Hook :obj:`~pyDeltaRCM.init_tools.init_tools.load_checkpoint`.

        Called immediately before
        :obj:`~pyDeltaRCM.init_tools.init_tools.load_checkpoint`.
        """
        pass

    def hook_run_one_timestep(self):
        """Hook :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.run_one_timestep`.

        Called immediately before
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.run_one_timestep`.
        """
        pass

    def hook_output_data(self):
        """Hook :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.output_data`.

        Called immediately before
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.output_data`.
        """
        pass

    def hook_apply_subsidence(self):
        """Hook :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.apply_subsidence`.

        Called immediately before
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.apply_subsidence`.
        """
        pass

    def hook_finalize_timestep(self):
        """Hook :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.finalize_timestep`.

        Called immediately before
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.finalize_timestep`.
        """
        pass

    def hook_output_checkpoint(self):
        """Hook :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.output_checkpoint`.

        Called immediately before
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.output_checkpoint`.
        """
        pass

    def hook_init_water_iteration(self):
        """Hook :obj:`~pyDeltaRCM.water_tools.water_tools.init_water_iteration`.

        Called immediately before
        :obj:`~pyDeltaRCM.water_tools.water_tools.init_water_iteration`.
        """
        pass

    def hook_run_water_iteration(self):
        """Hook :obj:`~pyDeltaRCM.water_tools.water_tools.run_water_iteration`.

        Called immediately before
        :obj:`~pyDeltaRCM.water_tools.water_tools.run_water_iteration`.
        """
        pass

    def hook_compute_free_surface(self):
        """Hook :obj:`~pyDeltaRCM.water_tools.water_tools.compute_free_surface`.

        Called immediately before
        :obj:`~pyDeltaRCM.water_tools.water_tools.compute_free_surface`.
        """
        pass

    def hook_finalize_water_iteration(self, iteration):
        """Hook :obj:`~pyDeltaRCM.water_tools.water_tools.finalize_water_iteration`.

        Called immediately before
        :obj:`~pyDeltaRCM.water_tools.water_tools.finalize_water_iteration`.
        """
        pass

    def hook_sed_route(self):
        """Hook :obj:`~pyDeltaRCM.sed_tools.sed_tools.sed_route`.

        Called immediately before
        :obj:`~pyDeltaRCM.sed_tools.sed_tools.sed_route`.
        """
        pass

    def hook_route_all_sand_parcels(self):
        """Hook :obj:`~pyDeltaRCM.sed_tools.sed_tools.route_all_sand_parcels`.

        Called immediately before
        :obj:`~pyDeltaRCM.sed_tools.sed_tools.route_all_sand_parcels`.
        """
        pass

    def hook_topo_diffusion(self):
        """Hook :obj:`~pyDeltaRCM.sed_tools.sed_tools.topo_diffusion`.

        Called immediately before
        :obj:`~pyDeltaRCM.sed_tools.sed_tools.topo_diffusion`.
        """
        pass

    def hook_route_all_mud_parcels(self):
        """Hook :obj:`~pyDeltaRCM.sed_tools.sed_tools.route_all_mud_parcels`.

        Called immediately before
        :obj:`~pyDeltaRCM.sed_tools.sed_tools.route_all_mud_parcels`.
        """
        pass

    def hook_compute_sand_frac(self):
        """Hook :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.compute_sand_frac`.

        Called immediately before
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.compute_sand_frac`.
        """
        pass
