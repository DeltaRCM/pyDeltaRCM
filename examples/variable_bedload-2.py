with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    mdl_muddy = VariableBedloadModel(f_bedload=0.3,
                                     out_dir=output_dir)
    mdl_sandy = VariableBedloadModel(f_bedload=0.7)