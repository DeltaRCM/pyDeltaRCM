import abc


class hook_tools(abc.ABC):
    """Tools defining various hooks in the model.

    For an explanation on model hooks, see the :doc:`User Guide
    </guides/user_guide>`.
    """

    def _check_deprecated_hooks(self):
        """Check for any old hooks that are defined.

        Sometimes hook names need to be deprecated, due to changing underlying
        model mechanics or anything else. This may mean that an old hook is
        *no longer called*, but there is no way to warn the user about this.
        Therefore, we enforce that no old hooks may be used as method names of
        subclassing models.

        This helper method is early in the `DeltaModel` `__init__` routine.
        """
        _deprecated_list = {'hook_sed_route': 'hook_route_sediment',
                            'hook_run_one_timestep': 'hook_solve_water_and_sediment_timestep'}
        for old_hook, new_hook in _deprecated_list.items():
            if hasattr(self, old_hook):
                raise AttributeError(
                    f'`{old_hook}` is deprecated, '
                    f'and has been replaced with `{new_hook}`.')

    def hook_import_files(self):
        """Hook :obj:`~pyDeltaRCM.init_tools.init_tools.import_files`.

        Called immediately before
        :obj:`~pyDeltaRCM.init_tools.init_tools.import_files`.

        This is the recommended hook to use for defining custom yaml parameters
        for a subclassed model. To allow the model to successfully load custom
        yaml parameters, expected types and default values need to be provided
        as dictionary key-value pairs to the model attribute,
        `subclass_parameters`.

        The expected format is:

        .. code::

            self.subclass_parameters['custom_param'] = {
                'type': ['expected_type'], 'default': default_value}  # noqa: E501

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

    def hook_solve_water_and_sediment_timestep(self):
        """Hook :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep`.

        Called immediately before
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep`.
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

    def hook_route_water(self):
        """Hook :obj:`~pyDeltaRCM.water_tools.water_tools.route_water`.

        Called immediately before
        :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.route_water`.
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

    def hook_route_sediment(self):
        """Hook :obj:`~pyDeltaRCM.sed_tools.sed_tools.route_sediment`.

        Called immediately before
        :obj:`~pyDeltaRCM.sed_tools.sed_tools.route_sediment`.
        """
        pass

    def hook_init_sediment_iteration(self):
        """Hook :obj:`~pyDeltaRCM.sed_tools.sed_tools.init_sediment_iteration`.

        Called immediately before
        :obj:`~pyDeltaRCM.sed_tools.sed_tools.init_sediment_iteration`.
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

    def hook_init_output_file(self):
        """Hook :obj:`~pyDeltaRCM.init_tools.init_tools.init_output_file`.

        Called immediately before
        :obj:`~pyDeltaRCM.init_tools.init_tools.init_output_file`.

        Expected format for entries to the `meta` dictionary nested within the
        `self._save_var_list` dictionary:

        .. code::

            self._save_var_list['meta']['new_meta_name'] = ['varname', 'units',
                                                            'type', (dimensions)]  # noqa: E501

        Expected format for time-varying metadata entries to the `meta`
        dictionary nested within the `self._save_var_list` dictionary:

        .. code::

            self._save_var_list['meta']['new_meta_attribute_name'] = [
                None, 'units', 'type', (dimensions)]  # noqa: E501

        .. note::

            For a vector of time-varying metadata, the dimension
            should be specified as ('total_time').

        Expected format for time varying grid entries as keys within the
        `self._save_var_list` dictionary:

        .. code::

            self._save_var_list['new_grid_name'] = ['varname', 'units',
                                                    'type', (dimensions)]

        """
        pass
