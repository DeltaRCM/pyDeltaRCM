import matplotlib.pyplot as plt

import pyDeltaRCM

with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    delta = pyDeltaRCM.DeltaModel(out_dir=output_dir)

    for _t in range(0, 5):
        delta.update()

    delta.finalize()


fig, ax = plt.subplots()
ax.imshow(delta.bed_elevation, vmax=-3)
plt.show()
