**************************
YAML Parameter Definitions
**************************

.. note::
   Currently incomplete/unfinished

Model Settings
==============

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

Model Domain Parameters
=======================

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

dx
--
Either an *integer* or a *float*.
This parameter specifies the length of the cell faces in the grid in **meters**.

L0_meters
---------
Either an *integer* or a *float*.
Length of the land adjacent to the inlet in **meters**.
This can also be thought of as the length of the inlet channel.

S0
--
Either an *integer* or a *float*.
This sets the characteristic slope for the delta topset.
This parameter is dimensionless.

itermax
-------

Np_water
--------
Represents the number of "parcels" to split the input water discharge into for
the reduced-complexity flow routing. This parameter must be an *integer*

u0
--
Either an *integer* or a *float*.

N0_meters
---------
Either an *integer* or a *float*.

h0
--
Either an *integer* or a *float*.

H_SL
----
Either an *integer* or a *float*.

SLR
---
Either an *integer* or a *float*.

Np_sed
------
Represents the number of "parcels" to split the input sediment discharge into for
the reduced-complexity sediment routing. This parameter must be an *integer*

f_bedload
---------
Either an *integer* or a *float*.

C0_percent
----------
Either an *integer* or a *float*.

Csmooth
-------
Either an *integer* or a *float*.

toggle_subsidence
-----------------

sigma_max
---------
Either an *integer* or a *float*.

start_subsidence
----------------
Either an *integer* or a *float*.

Output Settings
===============

save_eta_figs
-------------

save_stage_figs
---------------

save_depth_figs
---------------

save_discharge_figs
-------------------

save_velocity_figs
------------------

save_figs_sequential
--------------------

save_eta_grids
--------------

save_stage_grids
----------------

save_depth_grids
----------------

save_discharge_grids
--------------------

save_velocity_grids
-------------------

save_dt
-------

checkpoint_dt
-------------

save_strata
-----------

save_checkpoint
---------------

resume_checkpoint
-----------------

Reduced-Complexity Routing Parameters
=====================================

omega_sfc
---------
Either an *integer* or a *float*.

omega_flow
----------
Either an *integer* or a *float*.

Nsmooth
-------

theta_water
-----------
Either an *integer* or a *float*.

coeff_theta_sand
----------------
Either an *integer* or a *float*.

coeff_theta_mud
---------------
Either an *integer* or a *float*.

beta
----

sed_lag
-------
Either an *integer* or a *float*.

coeff_U_dep_mud
---------------
Either an *integer* or a *float*.

coeff_U_ero_mud
---------------
Either an *integer* or a *float*.

coeff_U_ero_sand
----------------
Either an *integer* or a *float*.

alpha
-----
Either an *integer* or a *float*.
