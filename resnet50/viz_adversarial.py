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



# FGSM Data
epsilon         = np.array([0,     0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])
orig_fgsm       = np.array([76.13, 25.328, 7.574, 2.98, 1.48, 0.892, 0.646, 0.52, 0.448, 0.366])
compressed_fgsm = np.array([71.426, 19.738, 5.28, 1.742, 0.774, 0.4, 0.248, 0.158, 0.108, 0.088])

# LL Data
orig_ll       = np.array([76.13, 67.668, 53.642, 40.098, 29.498, 21.314, 16.00, 11.34, 8.478, 6.524])
compressed_ll = np.array([71.426, 61.82, 46.114, 33.28, 23.68, 16.926, 12.53, 9.346, 6.944, 5.244])

fig, ax = plt.subplots(figsize=(8.1,7.9))
ax.plot(epsilon, orig_fgsm, color=Blue[4], linewidth=5, marker='o', markevery=1, markersize=10, label= 'FGSM Original')
#ax.plot(epsilon, compressed_fgsm, color=Blue[4], linewidth=3, marker='o', markevery=1, markersize=10, label= 'FGSM MINT-compressed')
ax.plot(epsilon, compressed_fgsm, color=Purpul[4], linestyle='--', linewidth=5, marker='o', markevery=1, markersize=10, label= 'FGSM MINT-compressed')

ax.plot(epsilon, orig_ll, color=Blue[4], linewidth=5, marker='^', markevery=1, markersize=10, label= 'LL Original')
#ax.plot(epsilon, compressed_ll, color=Blue[4], linewidth=3, marker='^', markevery=1, markersize=10, label= 'LL MINT-compressed')
ax.plot(epsilon, compressed_ll, color=Purpul[4], linestyle='--', linewidth=5, marker='^', markevery=1, markersize=10, label= 'LL MINT-compressed')

#ax.set_xlim([60,1030])
ax.legend()
ax.set_xlabel('Normalized $\epsilon$', fontsize=30)
ax.set_ylabel('Accuracy (%)', fontsize=30)
plt.legend(fontsize=20, loc= "best")
plt.title('ResNet50 (43.00$\%$) - ILSVRC2012', fontsize=22)
plt.tick_params(labelsize=25)
plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./adv_resnet50.pdf', format='pdf', dpi=1200)

