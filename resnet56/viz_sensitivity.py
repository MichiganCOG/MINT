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
layers          = ['conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8','conv9', 'conv10',
                   'conv11','conv12','conv13','conv14','conv15','conv16','conv17','conv18','conv19', 'conv20',
                   'conv21','conv22','conv23','conv24','conv25','conv26','conv27','conv28','conv29', 'conv30',
                   'conv31','conv32','conv33','conv34','conv35','conv36','conv37','conv38','conv39', 'conv40',
                   'conv41','conv42','conv43','conv44','conv45','conv46','conv47','conv48','conv49', 'conv50',
                   'conv51','conv52','conv53','conv54', 'conv55', 'linear1','linear2']
compression     = np.array([0., 0.195312,0.046875,0.066406,0.136719,0.042969,0.000000,0.171875,0.000000,0.136719,0.031250,0.152344,0.062500,0.191406,0.082031,0.000000,0.171875,0.015625,0.265625,0.000000,0.222656,0.348633,0.387695,0.635742,0.246094,0.381836,0.315430,0.671875,0.166992,0.267578,0.290039,0.471680,0.318359,0.385742,0.362305,0.416016,0.255859,0.000000,0.664795,0.797607,0.715088,0.786133,0.799561,0.889648,0.722900,0.735596,0.817139,0.826172,0.717773,0.814941,0.736572,0.633545,0.251953,0.000000,0.200439,0.000000,0.000000])

fig, ax = plt.subplots(figsize=(8.1,7.9))
ax.xaxis.grid(False)
ax.plot(np.arange(1, len(layers)+1), compression*100., color=Purpul[4], linewidth=3, marker='o', markevery=1, markersize=10, label= 'FGSM Original')
ax.plot(np.arange(1, len(layers)+1), compression*100., 'g*', markevery=[0, 15, 19, 37, 53, 55], markersize=20, label= 'Untouched')
ax.set_xticks(np.arange(1, len(layers)+1, 5))
ax.set_xlim([0.5,58])
#ax.legend()
ax.set_xlabel('Layers', fontsize=30)
ax.set_ylabel('Compression (%)', fontsize=30)
#plt.legend(fontsize=13, loc= "best")
plt.title('ResNet56 - CIFAR10', fontsize=30)
plt.tick_params(labelsize=25)
#plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./sens_resnet56.pdf', format='pdf', dpi=1200)

