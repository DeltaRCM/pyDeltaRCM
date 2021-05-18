Updating model boundary conditions
==================================

In implementing custom model subclasses, it is common to want to change boundary conditions throughout the model run (see :doc:`variable_bedload`, :doc:`variable_velocity`).
In some situations, we want to change a single variable, and see the effect of changing *only this variable*, in essence, pushing the model out of a dynamic equilibrium. 
Another possibility is that we would want to change the boundary conditions of the inlet, *but maintain the dynamic equilibrium*.

Let's create a `DeltaModel` class to demonstrate.

.. code::

    mdl = DeltaModel()


.. doctest::
    :hide:

    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     mdl = pyDeltaRCM.DeltaModel(out_dir=output_dir)

Now, we can see that by the default settings, after initialization, the model flow velocity is 1.0 m/s, flow depth is 5 m, and so the unit water discharge is 5 m2/s.

.. doctest::

    >>> mdl.u0
    1.0
    >>> mdl.h0
    5.0
    >>> mdl.qw0
    5.0

If after some number of model iterations, we wanted to change the inlet flow velocity to be 2.0 m/s, we could simply set this value directly.

.. doctest::

    >>> mdl.u0 = 2.0

But, now the model has been thrown out of equilibrium, where the unit water discharge no longer matches the product of the flow depth and flow velocity.

.. doctest::

    >>> mdl.u0
    2.0
    >>> mdl.h0
    5.0
    >>> mdl.qw0
    5.0

To remedy this, we need to use the :obj:`~pyDeltaRCM.init_tools.init_tools.create_boundary_conditions` method, which will reinitialize a number of fields, based on the current value of the inlet flow velocity.

.. doctest::

    >>> mdl.create_boundary_conditions()
    >>> mdl.qw0
    10.0

.. important::

    You are responsible for ensuring that boundary conditions are updated in the appropriate manner after changing certain model parameters. **You need to call the method to reinitialize boundary conditions yourself!**
