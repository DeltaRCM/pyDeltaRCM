Defining custom YAML parameters
===============================

.. currentmodule:: pyDeltaRCM.hook_tools

Custom subclasses for the standard ``DeltaModel`` may require additional or custom parameters not listed in the :doc:`/reference/model/yaml_defaults`.
By using the hook, :obj:`~hook_tools.hook_import_files`, it is straightforward to define custom YAML parameters along with expected types and default values for your subclassed model.

The following subclass model demonstrates this by defining a custom boolean parameter (could be used to toggle some custom functionality on/off), and a custom numeric parameter (could be required for the custom function).

.. doctest::

    >>> import pyDeltaRCM

    >>> class CustomParamsModel(pyDeltaRCM.DeltaModel):
    ...     """A subclass of DeltaModel with custom YAML parameters.
    ...
    ...     This subclass defines custom YAML parameters, their expected types
    ...     and default values.
    ...     """
    ...     def __init__(self, input_file=None, **kwargs):
    ...
    ...         # inherit base DeltaModel methods
    ...         super().__init__(input_file, **kwargs)
    ...
    ...     def hook_import_files(self):
    ...         """Define the custom YAML parameters."""
    ...         # custom boolean parameter
    ...         self.subclass_parameters['custom_bool'] = {
    ...             'type': 'bool', 'default': False
    ...         }
    ...
    ...         # custom numeric parameter
    ...         self.subclass_parameters['custom_number'] = {
    ...             'type': ['int', 'float'], 'default': 0
    ...         }

If the subclass model is loaded with a YAML configuration file that does not explicitly define these custom parameters, then the default values will be assigned as model attributes.

.. code::

    >>> defaults = CustomParamsModel()

.. doctest::
    :hide:

    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     defaults = CustomParamsModel(out_dir=output_dir)

.. doctest::

    >>> print(defaults.custom_bool)
    False

    >>> print(defaults.custom_number)
    0

.. note::

   Since custom YAML parameters have expected types, ``TypeErrors`` are raised if the custom parameter type provided in the YAML does not agree with what is expected as defined in the subclass.


Once the custom parameters have been defined in the subclassed model they can be treated just like the default model parameters, and can be specified in the YAML or as keyword arguments (``**kwargs``).

.. code::

    >>> customized = CustomParamsModel(custom_bool=True, custom_number=15.3)


.. doctest::
    :hide:

    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     customized = CustomParamsModel(out_dir=output_dir,
    ...                                    custom_bool=True,
    ...                                    custom_number=15.3)

.. doctest::

    >>> print(customized.custom_bool)
    True

    >>> print(customized.custom_number)
    15.3
