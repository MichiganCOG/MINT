""" Simple Code To Visualize Impact Of Group Size on Params Pruned """


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats

sns.set()
sns.set_context('paper')
sns.set_style("whitegrid")


current_palette = sns.color_palette()
current_palette = [[102,194,165],[252,141,98],[141,160,203]]
Green = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)
Purpul = sns.cubehelix_palette(8)
lightpurpul = sns.dark_palette("purple")
Blue = sns.cubehelix_palette(8, start=.5, rot=-.75)
Allcolor = sns.color_palette("hls", 8)
current_palette = [[i / 256.0 for i in color] for color in current_palette]



# Data
groups                 = ['5',   '10',   '20',   '50']
params_nolimit         = np.array([86.27, 87.25, 88.48, 91.87])
params_gamma_08        = np.array([86.54, 86.21, 76.96, 77.32])
params_gamma_linear    = np.array([11.96, 17.00, 38.05, 91.87])
params_gamma_quadratic = np.array([3.96, 3., 6.06, 91.87])

fig, ax = plt.subplots(figsize=(10,7.9))
ax.plot(groups, params_nolimit,  color=Blue[3], linewidth=5, linestyle='--', marker='o', markevery=1, markersize=20, label= 'No $\gamma$')
ax.plot(groups, params_gamma_08, color=Purpul[6], linewidth=5, linestyle='--', marker='^', markevery=1, markersize=20, label= '$\gamma = 0.8$')
ax.plot(groups, params_gamma_linear, color=Green[6], linewidth=5, linestyle='--', marker='*', markevery=1, markersize=20, label= '$\gamma =$ Linear')
ax.plot(groups, params_gamma_quadratic, color=Allcolor[6], linewidth=5, linestyle='--', marker='x', markevery=1, markersize=20, label= '$\gamma =$ Quadratic')

#pointer = ax.bar(groups,params, width=0.5, color=Purpul[4])
#for pointer_idx in range(len(pointer)):
#    ht = pointer[pointer_idx].get_height()
#    ax.text(pointer[pointer_idx].get_x() + pointer[pointer_idx].get_width()/15., 1.005*ht, '(%.2f)' % (acc[pointer_idx]), fontsize=20)

#ax.set_yticks(np.arange(80, 95, 0.01))
ax.set_ylim([0,93])
#ax.set_xlim([60,1030])
ax.legend()
ax.set_xlabel('Groups (G)', fontsize=30)
ax.set_ylabel('Params. Pruned (%)', fontsize=30)
#plt.legend(fontsize=25, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.legend(fontsize=25, loc="best")
plt.tick_params(labelsize=25)
plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./gamma.pdf', format='pdf', dpi=1200)

