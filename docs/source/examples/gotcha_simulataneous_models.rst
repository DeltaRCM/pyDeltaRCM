Instantiating and running two models iteratively on the same core is not reproducible
=====================================================================================


You cannot run models "simultaneously" and get reproducible results. 

In an ideal world, two models with same seed should give same result,
regardless of how they are run. But if two models are called intermittently,
then the result is not reproducible (at least on a per-model basis) because
the random-number generator is thrown out of sync with respect to the single
model and seed. More explanation below.

.. note::

    As of 03/2026, numba does not implement random number generation as an
    object, and so we cannot aviod this "gotcha".

.. code::
    
        # initialize models
        test1 = DeltaModel()
        test2 = DeltaModel()

        for _ in range(0, 5):
            test1.update()
            test2.update()

        test1.finalize()
        test2.finalize()

        # test that gives same result (WILL FAIL)
        difference = test1.eta - test2.eta
        assert np.all(difference == 0)

.. DEVNOTE: see test in tests/integration/test_consistent_outputs.py in TestConsistentOutputsSameSeed called test_same_models_simulataneous_not_parallel

**The above code will fail to generate reproducible results.**

The problem is that both test and test2 access the same underlying random
number generator in Numba, and therefore follow different trajectories over
time. The only way around this is to instantiate the models sequentially
(resetting the seed via the instantiation of the second one after finishing
the first one), or instantiating the objects in subprocesses (this is what
the `preprocessor` does for parallel runs). 

Of course, if you don't care about reproducibility at the moment (e.g., prototyping, testing approaches, etc), you can totally use the above approach. **Your final model code for analysis should be reproducible and not use this approach!** 


The below code will work to give reproducible results.

.. code::

    test1 = DeltaModel(input_file=p1)
    for _ in range(0, 5):
        test1.update()
        test1.finalize()

    test2 = DeltaModel(input_file=p2)
    for _ in range(0, 5):
        test2.update()
        test2.finalize()

