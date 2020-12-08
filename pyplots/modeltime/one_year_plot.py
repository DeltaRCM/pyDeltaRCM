import matplotlib.pyplot as plt
import numpy as np

import _base

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.2)
ax.plot(_base.time, _base.hydrograph)
ax.axhline(y=2.5, color='k', ls='--')
ax.fill_between(np.arange(365), 2.5 * np.ones((len(_base.time),)),
                _base.hydrograph, where=_base.flood, alpha=0.4)
ax.set_yticks([1.05, 2.5])
ax.set_yticklabels(['base flow', 'bankfull flow'])
ax.set_xlabel('calendar day')
ax.set_ylim(0.9, 3.1)

bbox = dict(boxstyle="round", fc="0.8")
_cs0 = "angle,angleA=-10,angleB=-60,rad=10"
ax.annotate('start event', xy=(95, 1.1), xytext=(0, 1.4),
            bbox=bbox, arrowprops=dict(arrowstyle="->", connectionstyle=_cs0))
_cs1 = "angle,angleA=10,angleB=60,rad=10"
ax.annotate('end event', xy=(150, 1.1), xytext=(170, 1.4),
            bbox=bbox, arrowprops=dict(arrowstyle="->", connectionstyle=_cs1))
_cs2 = "angle,angleA=-10,angleB=60,rad=10"
ax.annotate('flooding duration', xy=(110, 2.7), xytext=(150, 2.7),
            bbox=bbox, arrowprops=dict(arrowstyle="->", connectionstyle=_cs2))

plt.show()
