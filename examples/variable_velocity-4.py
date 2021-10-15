# we create the model here, just to be sure it works (for good docs)
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    mdl = ChangingVelocityModel(
        end_time=86400*100,
        out_dir=output_dir)