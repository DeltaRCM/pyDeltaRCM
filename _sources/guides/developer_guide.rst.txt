***************
Developer Guide
***************

.. image:: https://github.com/DeltaRCM/pyDeltaRCM/workflows/actions/badge.svg
    :target: https://github.com/DeltaRCM/pyDeltaRCM/actions

This guide provides additional details for implementation that did not make it into the user guide.
If you have not yet read the user guide, that is the place to start, then visit and refer to this guide for details as necessary.


==============
Best Practices
==============

Reproducibility
---------------

.. important:: tl;dr: any random numbers must be generated within a jitted function. 

A major goal of the `pyDeltaRCM` project is to enable fully reproducible simulation results.
Variability in pyDeltaRCM arises from the weighted random walks of water and sediment parcels during model iteration, so the state of the random source is essential to reproducibility.
We encourage developers to ensure that their subclass models are also reproducible, and provide some information on how to do so in this section.

For many subclassing models, it will be straightforward to ensure models are reproducible.
Out of the box, models will use the core `pyDeltaRCM` "seed" functionality to make models reproducible, and checkpointing should easily integrate with most use cases.
However, models that implement processes or functions that rely on random numbers or samples from random distributions will need to take care to ensure models are reproducible.

Random numbers
~~~~~~~~~~~~~~

For `pyDeltaRCM`, the source of random values is a pseudo-random number generator (RNG) from `numpy`, which works by inputting the current state of the RNG to an algorithm, and returning a sample (an integer) that can be mapped to any probability density function.
Each time the RNG yields a sample, the RNG state is changed, such that repeated samples from the RNG appear to be random, but are actually a deterministic sequence for a given initial RNG state (i.e., for a given seed).

As a result of the deterministic RNG, model runs are *exactly* reproducible if they begin from the same initial RNG state.
However, this means that any change to the state of the RNG that is not recorded will make a run non-reproducible. 
Additionally, any "random" behavior implemented in the model *that does not also* modify the state of the RNG, is non-reproducible. 

With a default setup of `numpy`, the functions of `np.random` will utilize the same underlying random number generator, such that the following code always evaluates to the same results.
In this case, it would be relatively simple to keep runs reproducible, because any call to a `np.random` function would modify the state of the underlying generator.

.. doctest::

    >>> np.random.seed(10); np.random.uniform(); np.random.normal(); np.random.uniform();
    0.771320643266746
    0.03777261440227079
    0.7488038825386119

In `pyDeltaRCM` though, we use `numba` just-in-time compilation for several steps in the model routine to make execution faster, including getting the random values that drive parcel movement during model iteration.
Within a Python console, calls to the `numba` RNG will not affect the state of the `numpy` RNG, and vice versa; even though the pre-JIT compiled code appears to call `np.random.uniform` (e.g., :obj:`~pyDeltaRCM.shared_tools.get_random_uniform`).

.. important:: `pyDeltaRCM` only takes responsibility for the `numba` RNG!


A simple example
~~~~~~~~~~~~~~~~

So why does it matter?
Well, calling `np.random.normal(100, 10)` in a model subclass will give you a random value, and modify the state of the `numpy` RNG, but it will not affect the state of the `numba` RNG (the one `pyDeltaRCM` actually keeps track of).
Thus, the values returned from the call to `np.random.normal` are not known and are not reproducible.

In the following simple example, see how the reproducible model uses a random number generated from a jitted function. This ensures the `numba` RNG is used for random variability in the model, and runs are reproducible.


.. doctest:: 

    >>> from numba import njit

    >>> @njit
    ... def get_random_normal():
    ...     """Get a random number from standard normal distribution.
    ...     """
    ...     return np.random.normal(0, 1)
    

    >>> class BrokenAndNotReproducible(pyDeltaRCM.DeltaModel):
    ... 
    ...     def __init__(self, input_file=None, **kwargs):
    ...         """Initialize a model that can never be reproduced.
    ...         """
    ... 
    ...         super().__init__(input_file=input_file, **kwargs)
    ... 
    ...     def update(self):
    ...         """Reimplement update method for demonstration."""
    ... 
    ...         # the core pyDeltaRCM RNG is used in computations, e.g.,
    ...         _sample0 = pyDeltaRCM.shared_tools.get_random_uniform(1)
    ... 
    ...         # now, we do something custom in our subclass
    ...         _sample1 = np.random.normal(0, 1)
    ... 
    ...         # and write it out to view
    ...         print(_sample0, _sample1)
    

    >>> class BeautifulAndVeryReproducible(pyDeltaRCM.DeltaModel):
    ... 
    ...     def __init__(self, input_file=None, **kwargs):
    ...         """Initialize a reproducible model.
    ...         """
    ... 
    ...         super().__init__(input_file=input_file, **kwargs)
    ... 
    ...     def update(self):
    ...         """Reimplement update method for demonstration."""
    ... 
    ...         # the core pyDeltaRCM RNG is used in computations, e.g.,
    ...         _sample0 = pyDeltaRCM.shared_tools.get_random_uniform(1)
    ... 
    ...         # now, we do something custom in our subclass
    ...         _sample1 = get_random_normal()
    ... 
    ...         # and write it out to view
    ...         print(_sample0, _sample1)

Now, we will initialize and run each model for three timesteps, twice. Running each twice will allow us to see if the model is reproducible (i.e., are all of the numbers exactly the same between runs).
First, run the `Broken` model:

.. doctest::
    :hide:

    # generate this for good docs, but it is not shown
    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     broken = BrokenAndNotReproducible(
    ...         out_dir=output_dir, seed=10)

.. code::

    broken = BrokenAndNotReproducible(seed=10)

.. doctest::

    >>> for i in range(3):
    ...     broken.update() # doctest: +SKIP
    0.771320643266746 1.213653088541954
    0.0207519493594015 -0.40009453994985783
    0.6336482349262754 0.7719410676752912

.. doctest::
    :hide:

    # generate this for good docs, but it is not shown
    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     broken = BrokenAndNotReproducible(
    ...         out_dir=output_dir, seed=10)

.. code::

    broken = BrokenAndNotReproducible(seed=10)

.. doctest::

    >>> for i in range(3):
    ...     broken.update() # doctest: +SKIP
    0.771320643266746 -0.17926017697487434
    0.0207519493594015 -0.4421037872728855
    0.6336482349262754 -0.2725394596633578

Now, run the reproducible model:

.. doctest::
    :hide:

    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     beautiful = BeautifulAndVeryReproducible(
    ...         out_dir=output_dir, seed=10)

.. code::

    beautiful = BeautifulAndVeryReproducible(seed=10)

.. doctest::

    >>> for i in range(3):
    ...     beautiful.update()
    0.771320643266746 0.03777261440227079
    0.7488038825386119 -0.1354484915560101
    0.4985070123025904 -0.6643797082723693


.. doctest::
    :hide:

    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     beautiful = BeautifulAndVeryReproducible(
    ...         out_dir=output_dir, seed=10)

.. code::

    beautiful = BeautifulAndVeryReproducible(seed=10)

.. doctest::

    >>> for i in range(3):
    ...     beautiful.update()
    0.771320643266746 0.03777261440227079
    0.7488038825386119 -0.1354484915560101
    0.4985070123025904 -0.6643797082723693

From these results, we can see that the values returned from the built-in uniform RNG as the first sample of each iteration (i.e., the left column) is always deterministic (in `broken` and `beautiful`), whereas both the built-in and the custom RNG are deterministic (in `beautiful`).


.. important::
    
    Be sure to only generate random numbers inside jitted functions!

.. note::

    It is generally okay to not worry about reproducibility when you are developing your subclassing model and trying to work out how model mechanics will depend on randomness -- but once you start to do real simulations you may analyze, be sure to take the time to make your model reproducible.



Model development
-----------------

Slicing and neighbors 
~~~~~~~~~~~~~~~~~~~~~

Slicing an array to find the array values of neighbors is a common operation in the model.
The preferred way to slice is by 1) padding the array with :func:`~pyDeltaRCM.shared_tools.custom_pad`, and 2) looping through rows and columns to directly index. This approach makes for readable and reasonably fast code; for example, to find any cells that are higher than all neighbors:

.. code::
    
    pad_eta = shared_tools.custom_pad(self.eta)
    for i in range(self.L):
        for j in range(self.W):
            eta_nbrs = pad_eta[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
            eta_nbrs[1, 1] = -np.inf
            
            np.all(self.eta[i, j] > eta_nbrs)

There are also several model attributes that may be helpful in development; we suggest using these builtins rather than creating your own whenever possible (see :meth:`~pyDeltaRCM.init_tools.init_tools.set_constants` and the model source code).
