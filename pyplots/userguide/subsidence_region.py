import matplotlib.pyplot as plt
from math import pi
import pyDeltaRCM

delta = pyDeltaRCM.DeltaModel()
delta.toggle_subsidence = True
delta.theta1 = -pi/2
delta.theta2 = 0
delta.init_subsidence()


fig, ax = plt.subplots()
ax.imshow(delta.subsidence_mask)
plt.show()
