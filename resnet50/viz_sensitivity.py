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
                   'conv51','conv52','conv53', 'linear1']
compression     = np.array([0.000000,0.734619,0.319092,0.441406,0.343750,0.031250,0.285156,0.270264,0.175781,0.265869,0.352783,0.485107,0.278564,0.361328,0.833252,0.245605,0.290283,0.337646,0.438477,0.517578,0.141846,0.382324,0.329834,0.329834,0.847656,0.357422,0.813232,0.774414,0.107422,0.120605,0.529785,0.466797,0.908447,0.885254,0.669434,0.258057,0.210938,0.486328,0.223145,0.041260,0.324463,0.282471,0.301514,0.034668,0.052246,0.910645,0.842285,0.673828,0.616455,0.520264,0.891113,0.779541,0.684570,0.000000])

fig, ax = plt.subplots(figsize=(8.1,7.9))
ax.xaxis.grid(False)
ax.plot(np.arange(1, len(layers)+1), compression*100., color=Purpul[4], linewidth=3, marker='o', markevery=1, markersize=10, label= 'FGSM Original')
ax.plot(np.arange(1, len(layers)+1), compression*100., 'g*', markevery=[0, 53], markersize=20, label= 'Untouched')
ax.set_xticks(np.arange(1, len(layers)+1, 5))
ax.set_xlim([0.5,56])
#ax.legend()
ax.set_xlabel('Layers', fontsize=30)
ax.set_ylabel('Compression (%)', fontsize=30)
#plt.legend(fontsize=13, loc= "best")
plt.title('ResNet50 - ILSVRC2012', fontsize=30)
plt.tick_params(labelsize=25)
#plt.setp(ax.spines.values(), linewidth=1, color ='k')
fig.savefig('./sens_resnet50.pdf', format='pdf', dpi=1200)

