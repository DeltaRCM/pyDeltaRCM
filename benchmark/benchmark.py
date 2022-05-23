import cProfile
import pstats
import time
from datetime import datetime

import pyDeltaRCM

delta = pyDeltaRCM.DeltaModel(save_eta_figs=False)

start_time = 0.0
profiler = cProfile.Profile()
for i in range(10):
   if i==3:
      profiler.enable()
      start_time = time.time()

   print(f"Iteration {i}")
   delta.update()

end_time = time.time()

profiler.disable()
profiler.print_stats()

print(f"Total time = {end_time - start_time}")

my_date = datetime.now()
stats = pstats.Stats(profiler)
stats.dump_stats(f"bench_result_{my_date.isoformat()}.prof")
delta.finalize()

# Results can be plotted with, eg, `flameprof -o /z/out.svg bench_result_2022-05-23T10:39:22.808217.prof`