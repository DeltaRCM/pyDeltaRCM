import matplotlib.pyplot as plt
import numpy as np


import _base


fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
# plt.subplots_adjust(top=0.7)


day = 0
for i in range(4):
    ax[0].axvline(x=day, color='k', ls='-')
    ax[0].fill_between(day + np.arange(365), np.zeros((len(_base.time),)),
                    4 * np.ones((len(_base.time),)),
                    where=_base.flood, alpha=0.4, edgecolor='none')
    ax[0].plot(day + np.arange(365), _base.hydrograph)
    ax[0].text(day + 220, 2.6, 'year ' + str(i))
    day += 365

ax[0].axvline(x=day, color='k', ls='-')
ax[0].set_ylim(0.9, 3.1)
ax[0].set_xlim(0 - 10, day + 10)
ax[0].set_xlabel(r'time $\rightarrow$')
ax[0].set_yticks([1.05, 2.5])
ax[0].set_yticklabels(['base flow', 'bankfull flow'])


nflood = np.sum(_base.flood)
day = 0
for i in range(4):
    ax[1].axvline(x=day, color='k', ls='-')
    ax[1].fill_between(day + np.arange(nflood + 1), np.zeros((nflood + 1,)),
                    4 * np.ones((nflood + 1,)),
                    alpha=0.4, edgecolor='none')
    ax[1].plot(day + np.arange(nflood + 1), 2.5 * np.ones((nflood + 1),))
    day += nflood

ax[1].axvline(x=day, color='k', ls='-')
ax[1].set_ylim(0.9, 3.1)
ax[1].set_xlim(0 - 10, 365 * 4 + 10)
ax[1].set_yticks([])
ax[1].set_xlabel(r'time $\rightarrow$')
ax[1].set_yticks([1.05, 2.5])
ax[1].set_yticklabels(['base flow', 'bankfull flow'])


plt.show()
