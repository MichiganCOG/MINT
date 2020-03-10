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
groups = ['5',    '10',   '20', '50']
acc    = [98.55, 98.52, 98.55, 98.55]
params = np.array([86.27, 87.25, 88.48, 91.87])

fig, ax = plt.subplots(figsize=(8.1,7.9))
#ax.plot(groups, params, color=Purpul[4], linewidth=3, marker='o', markevery=1, markersize=10, label= 'Params. Pruned (%)')

#ax.plot(groups, params, color=Blue[6], linewidth=3, linestyle='--', marker='o', markevery=1, markersize=10, label= 'Params. Pruned (%)')
pointer = ax.bar(groups,params, width=0.5, color=Purpul[4])
x_collect = []
for pointer_idx in range(len(pointer)):
    ht = pointer[pointer_idx].get_height()
    ax.text(pointer[pointer_idx].get_x() + pointer[pointer_idx].get_width()/15., 1.005*ht, '(%.2f)' % (acc[pointer_idx]), fontsize=20)
    x_collect.append(pointer[pointer_idx].get_x()+ pointer[pointer_idx].get_width()/2.)

ax.set_ylim([86,94])
#ax.legend()
ax.set_xlabel('Number of groups (G)', fontsize=30)
ax.set_ylabel('Params. Pruned (%)', fontsize=30)
#plt.legend(fontsize=13, loc= "best")
plt.tick_params(labelsize=25)
plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./groups.pdf', format='pdf', dpi=1200)

