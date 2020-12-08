import matplotlib.pyplot as plt
import numpy as np


import _base


fig, ax = plt.subplots(figsize=(8, 2))
plt.subplots_adjust(top=0.7)

nflood = np.sum(_base.flood)
day = 0
for i in range(4):
    ax.axvline(x=day, color='k', ls='-')
    ax.fill_between(day + np.arange(nflood + 1), np.zeros((nflood + 1,)),
                    4 * np.ones((nflood + 1,)),
                    alpha=0.4, edgecolor='none')
    ax.plot(day + np.arange(nflood + 1), 2.5 * np.ones((nflood + 1),))
    day += nflood

ax.axvline(x=day, color='k', ls='-')
ax.set_ylim(0.9, 3.1)
ax.set_xlim(0 - 10, 365 * 4 + 10)
ax.set_yticks([])
ax.set_xlabel(r'time $\rightarrow$')
ax.set_yticks([1.05, 2.5])
ax.set_yticklabels(['base flow', 'bankfull flow'])

plt.show()
