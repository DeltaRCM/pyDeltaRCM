import matplotlib.pyplot as plt

import pyDeltaRCM

with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    delta = pyDeltaRCM.DeltaModel(out_dir=output_dir)

delta.show_attribute('cell_type', grid=False)
delta.show_ind([144, 22, 33, 34, 35])
delta.show_ind((12, 14), 'bs')
delta.show_ind([(11, 4), (11, 5)], 'g^')
plt.show()
