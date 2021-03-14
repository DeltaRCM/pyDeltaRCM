import matplotlib.pyplot as plt

import pyDeltaRCM

with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    delta = pyDeltaRCM.DeltaModel(out_dir=output_dir)

delta.show_attribute('cell_type', grid=False)
delta.show_attribute('cell_type', grid=False)
delta.show_ind([3378, 9145, 11568, 514, 13558])
delta.show_ind((42, 94), 'bs')
delta.show_ind([(41, 8), (42, 10)], 'g^')
plt.show()
