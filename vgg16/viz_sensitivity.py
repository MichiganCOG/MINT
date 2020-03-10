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
layers            = ['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight', 'conv10.weight','conv11.weight','conv12.weight','conv13.weight', 'linear1.weight', 'linear3.weight']
layers_untouched  = ['conv1.weight','conv2.weight','conv3.weight','conv4.weight', 'linear3.weight']
compression     = np.array([0.0, 0.0, 0.0, 0.0, 0.670898, 0.518066,0.583984, 0.611816, 0.899170, 0.937256, 0.920410, 0.862549, 0.910156, 0.932861, 0.0])

fig, ax = plt.subplots(figsize=(8.1,7.9))
ax.xaxis.grid(False)
ax.plot(np.arange(1, len(layers)+1), compression*100., color=Purpul[4], linewidth=3, marker='o', markevery=1, markersize=10, label= 'FGSM Original')
ax.plot(np.arange(1, len(layers)+1), compression*100., 'g*', markevery=[0,1,2,3,14], markersize=20, label='Untouched')
#ax.set_xticks(np.arange(1, len(layers)+1, 5))
ax.set_xlim([0,18])
#ax.legend()
ax.set_xlabel('Layers', fontsize=30)
ax.set_ylabel('Compression (%)', fontsize=30)
#plt.legend(fontsize=13, loc= "best")
plt.title('VGG16 - CIFAR10', fontsize=30)
plt.tick_params(labelsize=25)
#plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./sens_vgg16.pdf', format='pdf', dpi=1200)

