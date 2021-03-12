with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    mdl = ConstrainedSubsidenceModel(toggle_subsidence=True,
                                     out_dir=output_dir)