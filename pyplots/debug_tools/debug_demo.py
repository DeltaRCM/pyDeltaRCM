import matplotlib.pyplot as plt

import pyDeltaRCM


delta = pyDeltaRCM.DeltaModel()


delta.show_attribute('cell_type')
delta.show_ind([144, 22, 33, 34, 35])
delta.show_ind((12, 14), 'bs')
delta.show_ind([(11, 4), (11, 5)], 'g^')
plt.show()
