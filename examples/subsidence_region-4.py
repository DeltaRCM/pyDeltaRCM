from pyDeltaRCM.preprocessor import scale_relative_sea_level_rise_rate

subsidence_mmyr = np.array([3, 6, 10, 25, 50, 100])
subsidence_scaled = scale_relative_sea_level_rise_rate(subsidence_mmyr, If=0.019)