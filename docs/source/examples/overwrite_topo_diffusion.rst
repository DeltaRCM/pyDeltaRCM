Changing topographic diffusion to represent tree throw
======================================================

Here, we demonstrate how to overwrite an existing method of the `DeltaModel` to achieve custom behavior during model runtime.

.. note::

    This example demonstrates several best-practices, including using yaml parameters specifications, custom figure and grid saving setup, and using :obj:`~pyDeltaRCM.shared_tools.get_random_uniform` for random number generation.

In implementing custom model subclasses, it is common to want to change runtime behavior of the model. 
This can often be achieved by using hooks alone, but sometimes a combination of hooks and overwriting existing methods is necessary.

In this example, we calculate a diffusion multiplier to represent tree throw. 
In this super simple and **likely way over-exaggerated** representation of this process, we assume:

* there are trees everywhere the elevation of a cell has been above `self.dry_depth` for 10 consecutive timesteps
* there is a probability threshold of tree throw occurring anywhere trees exist
* tree throw makes the diffusion multiplier for that cell on that timestep really big!

.. important::

    There are all kinds of problems with the assumptions in this example. Don't read into it too much. It's an example to show how to modify code, not how to represent tree throw.

.. plot::
    :context: reset
    :include-source:

    class TreeThrowModel(pyDeltaRCM.DeltaModel):
        """Implementation of tree throw.
        """

        def __init__(self, input_file=None, **kwargs):

            # inherit from base model
            super().__init__(input_file, **kwargs)
            self.hook_after_create_domain()

        def hook_import_files(self):
            """Define the custom YAML parameters."""
            # whether to run vegetation
            self.subclass_parameters['tree_throw'] = {
                'type': 'bool', 'default': False
            }

            # tree throw multiplier
            self.subclass_parameters['p_tt_mult'] = {
                'type': ['int', 'float'], 'default': 100
            }

            # tree throw establish timesteps
            self.subclass_parameters['p_tt_iter'] = {
                'type': ['int', 'float'], 'default': 10
            }

            # tree throw prob threshold
            self.subclass_parameters['p_tt_prob'] = {
                'type': ['int', 'float'], 'default': 0.2
            }

        def hook_init_output_file(self):
            """Add non-standard grids, figures and metadata to be saved."""
            # save a figure of the active layer each save_dt
            self._save_fig_list['tree'] = ['tree']

            # save the active layer grid each save_dt w/ a short name
            self._save_var_list['tree'] = ['tree', 'boolean',
                                           'i4', ('time',
                                                  'x', 'y')]

        def hook_after_create_domain(self):
            """Add fields to the model for all tree parameterizations.
            """
            self.tree = np.zeros(self.depth.shape, dtype=np.int64)
            self.dry_count = np.zeros(self.depth.shape, dtype=np.int64)

            self.tree_multiplier = np.ones_like(self.depth)

        def hook_after_route_sediment(self):
            """Apply vegetation growth/death rules.
            """
            # determine cells dry and increment counter
            _where_dry = (self.depth < self.dry_depth)
            self.dry_count[~_where_dry] = 0  # any wet gets reset
            self.dry_count[_where_dry] += 1  # any dry gets incremented

            # if tree_throw is on, run the tree placing routine
            if self.tree_throw:

                # trees die anywhere wet
                self.tree[~_where_dry] = int(0)

                # trees go anywhere dry for more than threshold
                _where_above_thresh = (self.dry_count >= self.p_tt_iter)

                self.tree[_where_above_thresh] = int(1)

                # determine the multiplier field
                _rand = np.array([get_random_uniform(1) for i in np.arange(self.depth.size)]).reshape(self.depth.shape)
                _thrown = np.logical_and((_rand < self.p_tt_prob), self.tree)

                # ignore the strip of land
                _thrown[self.cell_type == -2] = 0

                # set to ones everywhere, then overwrite with multiplier
                self.tree_multiplier[:] = 1
                self.tree_multiplier[_thrown] = self.p_tt_mult

        def topo_diffusion(self):
            """Overwrite with new behavior.
        
            This method is very similar to the base DeltaModel code, but we add an
            additional multiplier to represent tree throw.
            """
            for _ in range(self.N_crossdiff):

                a = ndimage.convolve(self.eta, self.kernel1, mode='constant')
                b = ndimage.convolve(self.qs, self.kernel2, mode='constant')
                c = ndimage.convolve(self.qs * self.eta, self.kernel2,
                                     mode='constant')

                self.cf = (self.tree_multiplier * self.diffusion_multiplier *
                           (self.qs * a - self.eta * b + c))

                self.cf[self.cell_type == -2] = 0
                self.cf[0, :] = 0

                self.eta += self.cf

And the model is then instantiated with:

.. plot::
    :context:

    with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
        mdl = TreeThrowModel(
            out_dir=output_dir,
            tree_throw=True)

.. code:: python

    mdl = TreeThrowModel(
        tree_throw=True)

We don't actually run this model at all in this example.
Let's plot the ``.tree`` field just to see that the subclass was instantiated correctly.

.. plot::
    :context:
    :include-source:

    fig, ax = plt.subplots()
    im = ax.imshow(mdl.trees)
    plt.colorbar(im, ax=ax, shrink=0.5)
    plt.show()
