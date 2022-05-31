Basic script to run the model
=============================

.. plot::
    :context: reset

    with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
        delta = pyDeltaRCM.DeltaModel(
            out_dir=output_dir,
            resume_checkpoint='../_resources/checkpoint')
        delta.finalize()


.. code::

    delta = pyDeltaRCM.DeltaModel()

    for _t in range(0, 1000):
        delta.update()

    delta.finalize()

.. plot::
    :context:
    :include-source:

    fig, ax = plt.subplots()
    ax.imshow(delta.eta)
    plt.show()
