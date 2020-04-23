import sys
import os
sys.path.append(os.path.realpath(os.getcwd() + "/../../../../"))

import matplotlib.pyplot as plt

import pyDeltaRCM


delta = pyDeltaRCM.pyDeltaRCM()

for _t in range(0, 3):
    delta.update()

delta.finalize()


fig, ax = plt.subplots()
ax.imshow(delta.bed_elevation)
plt.show()
