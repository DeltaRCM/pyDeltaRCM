with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    param_dict['out_dir'] = output_dir
    pp = pyDeltaRCM.Preprocessor(
        param_dict,
        parallel=False,
        timesteps=1)
    pp.run_jobs(DeltaModel=ConstrainedSubsidenceModel)