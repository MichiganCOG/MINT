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
groups = ['150',   '250',   '450',   '650']
acc    = [98.58, 98.53, 98.51,  98.53]
params = np.array([85.35, 88.48, 88.72, 89.70])

fig, ax = plt.subplots(figsize=(9.0,9.0))
#ax.plot(groups, params, color=Purpul[6], linewidth=3, linestyle='--', marker='o', markevery=1, markersize=10, label= 'Params. Pruned (%)')

pointer = ax.bar(groups,params, width=0.5, color=Purpul[4])
for pointer_idx in range(len(pointer)):
    ht = pointer[pointer_idx].get_height()
    ax.text(pointer[pointer_idx].get_x() - pointer[pointer_idx].get_width()/10., 1.005*ht, '(%.2f)' % (acc[pointer_idx]), fontsize=25)

#ax.set_yticks(np.arange(80, 95, 0.01))
ax.set_ylim([85,91])
#ax.set_xlim([60,1030])
#ax.legend()
ax.set_xlabel('Samples per class (m)', fontsize=35)
ax.set_ylabel('Params. Pruned (%)', fontsize=35)
#plt.legend(fontsize=13, loc= "best")
plt.tick_params(labelsize=25)
plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./samples.png', format='png', dpi=1200)

