# we create the model here, just to be sure it works (for good docs)
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    mdl = ChangingVelocityModel(
        end_time=end_time,
        out_dir=output_dir)