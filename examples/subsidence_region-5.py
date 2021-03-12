# add a matrix with subsidence to the dict
param_dict = {}
param_dict['matrix'] = {'subsidence_rate': subsidence_scaled}

# add other configurations
param_dict.update(
    {'out_dir': 'liang_2016_reproduce',
     'toggle_subsidence': True,
     'parallel': 3})  # we can take advantage of parallel jobs