import sys
import os
sys.path.append(os.path.realpath(os.getcwd() + "/../../../../"))

import matplotlib.pyplot as plt

import pyDeltaRCM


delta = pyDeltaRCM.DeltaModel()

for _t in range(0, 1):
    delta.update()

delta.finalize()


fig, ax = plt.subplots()
ax.imshow(delta.bed_elevation, vmax=-4.5)
plt.show()
