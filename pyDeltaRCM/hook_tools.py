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

    def hook_init_grids(self):
        """
        Hook within :obj:`~pyDeltaRCM.init_tools.init_tools.init_output_file`.

        Called within :obj:`~pyDeltaRCM.init_tools.init_tools.init_output_file`
        after establishment of the output netCDF4 file and its dimensions.
        But before creation of standard grids in the netCDF file. Look at the
        function :obj:`~pyDeltaRCM.init_tools.init_tools.init_output_file` for
        guidance on how to write this hook to set up additional netCDF grids.
        """
        pass

    def hook_init_metadata(self):
        """
        Hook within :obj:`~pyDeltaRCM.init_tools.init_tools.init_output_file`.

        Called within :obj:`~pyDeltaRCM.init_tools.init_tools.init_output_file`
        after creation of the netCDF4 metadata group, 'meta'.
        But before assignment of standard metadata variables. Look at the
        function :obj:`~pyDeltaRCM.init_tools.init_tools.init_output_file` for
        guidance on how to write this hook to set up additional metadata.
        """
        pass

    def hook_save_grids(self, save_idx):
        """
        Hook within :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.save_grids_and_figs`.

        Called within
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.save_grids_and_figs`
        before standard grids are saved to the netCDF file.
        """
        pass

    def hook_save_metadata(self, save_idx):
        """
        Hook within :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.save_grids_and_figs`.

        Called within
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.save_grids_and_figs`
        before the standard time-varying metadata is saved to the group.
        """
        pass
