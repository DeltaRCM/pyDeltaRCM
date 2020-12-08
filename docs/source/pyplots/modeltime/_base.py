import numpy as np

time = np.arange(365)
hydrograph = np.ones((len(time),)) + \
    (np.random.uniform(size=(len(time),)) / 10)
hydrograph[100:150] = 2 + np.sin(np.arange(15, 65) / 15)
flood = hydrograph >= 2.5
