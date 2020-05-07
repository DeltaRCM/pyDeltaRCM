import sys
import os
import time

import cProfile
import pstats

import numpy as np
import matplotlib.pyplot as plt

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM

# fix the random seed manually here, since we use the
#    defaults for model instantiation
np.random.seed(0)
delta = pyDeltaRCM()

# run these functions once so that numba can compile them to machine code
#   this is a "startup cost", but it only happens once,
#   then the benefit is reaped over every subsequent iteration.
delta.init_water_iteration()
delta.run_water_iteration()

# begin timing,
#   -- for ten iterations of the *whole* model update()
start = time.time()
for _t in range(0, 10):
    print("timestep:", _t)
    delta.update()
end = time.time()
# end timing

print("Elapsed (without compilation) = %s min" % str((end - start) / 60))


# run the profiler for one more update()
cProfile.run('delta.update()', '../temp/updatetime')
p = pstats.Stats('../temp/updatetime')
p.strip_dirs().sort_stats('cumtime').print_stats()
