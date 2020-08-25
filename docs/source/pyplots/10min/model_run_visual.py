import matplotlib.pyplot as plt

import pyDeltaRCM


delta = pyDeltaRCM.DeltaModel()

for _t in range(0, 5):
    delta.update()

delta.finalize()


fig, ax = plt.subplots()
ax.imshow(delta.bed_elevation, vmax=-3)
plt.show()
