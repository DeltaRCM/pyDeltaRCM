***************
YAML Parameters
***************

Configurable model parameters are listed in the
:doc:`../reference/model/yaml_defaults`, below are descriptions for each
parameter.

.. note::
   Currently incomplete/unfinished

out_dir
-------
A *string* type parameter, specifying the name of the output directory in which
the model outputs should be saved.

verbose
-------
An *integer* type parameter, which when set to 1 will generate a full log of
messages and warnings as the model is run. When set to 0 (the default), many of
the model messages and warnings are suppressed.

seed
----
An *integer* type parameter specifying the random seed value to be used for
this model run. If unspecified, a random seed is generated and used.

Length
------
Either an *integer* or a *float*.
This is the length of the domain (dimension parallel to the inlet channel), in
**meters**.

Width
-----
Either an *integer* or a *float*.
This is the width of the domain (dimension perpendicular to the inlet channel),
in **meters**.
