Saving custom fields to the output file
=======================================

Given the flexibility of the pyDeltaRCM model to modifications via hooks and subclassing, it is necessary that the variables saved to the output netCDF file are similarly customizable.
Fortunately, the use of subclasses and hooks itself enables flexible setting of gridded variables to be saved as figures, as well as customization of the fields saved to the netCDF file as both variables, and as metadata.

To customize the figures and variables that are saved, the hook ``hook_init_output_file`` should be used to subclass the pyDeltaRCM model.

When adding a model attribute to the key-value pairs of grids to save as figures, the key indicates the name of the figure file that will be saved, and the value-pair can be a string representing the model attribute to be plotted, or a combination of model attributes, such as ``self.eta * self.depth``.
For example, ``self._save_fig_list['active_layer'] = ['active_layer']`` will properly indicate that figures of the active layer should be saved.

.. important::

    The built-in, on-the-fly, figure saving as the model runs is only supported for gridded variables with the shape ``L x W`` matching the model domain.
    Trying to set up figures to save that are not variables of that shape will result in an error.

When adding variables or metadata to be initialized and subsequently saved in the output netCDF, the key-value pair relationship is as follows.
The key added to ``self._save_var_list`` is the name of the variable as it will be recorded in the netCDF file, this *does not* have to correspond to the name of an attribute in the model.
To add a variable to the metadata, a key must be added to ``self._save_var_list['meta']``.
The expected value for a given key is a list containing strings indicating the model attribute to be saved, its units, the variable type, and lastly the variable dimensions (e.g., ``['active_layer', 'fraction', 'f4', ('time', 'x', 'y')]`` for the active layer).

.. important::

    The dimensions of the custom variable being specified must match *exactly* with one of the three standard dimensions: `x`, `y`, `time`.
    Use of an invalid dimension will result in an error.

An example of using the hook and creating a model subclass to customize the figures, gridded variables, and metadata being saved is provided below.

.. doctest::

    >>> import pyDeltaRCM

    >>> class CustomSaveModel(pyDeltaRCM.DeltaModel):
    ...     """A subclass of DeltaModel to save custom figures and variables.
    ...
    ...     This subclass modifies the list of variables and figures used to
    ...     initialize the netCDF file and save figures and grids before the
    ...     output file is setup and initial conditions are plotted.
    ...     """
    ...     def __init__(self, input_file=None, **kwargs):
    ...
    ...         # inherit base DeltaModel methods
    ...         super().__init__(input_file, **kwargs)
    ...
    ...     def hook_init_output_file(self):
    ...         """Add non-standard grids, figures and metadata to be saved."""
    ...         # save a figure of the active layer each save_dt
    ...         self._save_fig_list['active_layer'] = ['active_layer']
    ...
    ...         # save the active layer grid each save_dt w/ a short name
    ...         self._save_var_list['actlay'] = ['active_layer', 'fraction',
    ...                                          'f4', ('time',
    ...                                                 'x', 'y')]
    ...
    ...         # save number of water parcels w/ a long name
    ...         self._save_var_list['meta']['water_parcels'] = ['Np_water',
    ...                                                         'parcels',
    ...                                                         'i8', ()]

Next, we instantiate the model class.

.. code::

    >>> mdl = CustomSaveModel()


.. doctest::
    :hide:

    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     mdl = CustomSaveModel(out_dir=output_dir)


This subclass has added the active layer as a figure and a grid to be saved, as well as the number of water parcels as metadata to be saved.
For simplicity we will just check that the appropriate parameters were added to the save figure and save variable lists, however please feel free to give this example a try on your local machine and examine the output figures and netCDF file.

.. doctest::

    >>> 'active_layer' in mdl._save_fig_list
    True

    >>> print(mdl._save_fig_list)
    {'active_layer': ['active_layer']}

    >>> print(mdl._save_var_list)
    {'meta': {'water_parcels': ['Np_water', 'parcels', 'i8', ()]}, 'actlay': ['active_layer', 'fraction', 'f4', ('time', 'x', 'y')]}
