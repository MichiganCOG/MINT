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
orig_fgsm       = np.array([93.98, 85.87, 75.59, 68.31, 62.74, 57.43, 52.53, 47.59, 43.08, 38.91])
compressed_fgsm = np.array([93.43, 81.88, 69.04, 58.35, 49.56, 43.12, 37.64, 33.27, 30.11, 27.26])

# LL Data
orig_ll       = np.array([93.98, 90.35, 84.84, 81.70, 78.47, 75.80, 72.56, 69.58, 66.57, 62.95])
compressed_ll = np.array([93.43, 88.46, 80.16, 73.41, 67.74, 62.30, 57.51, 53.76, 49.74, 46.82])

fig, ax = plt.subplots(figsize=(8.1, 7.9))
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
plt.title('VGG16 - CIFAR10', fontsize=22)
plt.tick_params(labelsize=25)
plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./adv_vgg16.pdf', format='pdf', dpi=1200)

