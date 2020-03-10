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
orig_fgsm       = np.array([92.55, 75.42, 53.68, 35.86, 23.32, 15.21, 9.89,  6.30,  4.24,  2.92 ])
compressed_fgsm = np.array([93.47, 71.84, 46.99, 28.55, 16.87, 10.26, 6.20,  4.02,  2.49,  1.52 ])

# LL Data
orig_ll       = np.array([92.55, 88.45, 80.25, 71.28, 62.65, 53.28, 44.78, 37.10, 30.37, 24.87 ])
compressed_ll = np.array([93.47, 89.02, 79.32, 68.53, 59.65, 49.86, 41.61, 34.41, 28.68, 23.56 ])

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
plt.title('ResNet56 (52.41$\%$) - CIFAR10', fontsize=22)
plt.tick_params(labelsize=25)
plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./adv_resnet56.pdf', format='pdf', dpi=1200)

