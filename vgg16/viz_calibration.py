import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

num_bins = 10

bin_upper = 1./num_bins*(1+np.array([*range(num_bins)]))
# don't know if this is the best way to do it, it'll run fast enough
# regardless
bin_values = []
bin_correct = []
for i in range(num_bins):
    bin_values.append([])
    bin_correct.append([])
with open(sys.argv[1],'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        for i in range(num_bins):
            if float(row[0]) < bin_upper[i]:
                bin_values[i].append(float(row[0]))
                bin_correct[i].append(float(row[1]))
means = []
correct_pct = []
bin_counts = []
for i in range(num_bins):
    means.append(np.mean(bin_values[i]))
    correct_pct.append(np.mean(bin_correct[i]))
    bin_counts.append(len(bin_correct[i]))
weights = np.array(bin_counts)/np.array(bin_counts).sum()
diff = np.abs(np.array(means) - np.array(correct_pct))
ece = np.nansum(weights*diff)
print("ECE is: " + str(ece))
##fig = plt.figure()
#fig, ax = plt.subplots(figsize=(8.1,7.5))
fig, ax = plt.subplots(figsize=(9.5,9.5))
#ax = fig.add_subplot(111)
#plt.bar(bin_upper-(1./num_bins),correct_pct,width=1./num_bins, color=Blue[4])
plt.bar(bin_upper-(1./num_bins),correct_pct,width=1./num_bins, color=Purpul[4])
line = plt.Line2D((0,1),(0,1), c='red')
#title_name =  'Orig. MLP ECE: %.4f'%(ece)
#title_name =  'MINT MLP ECE: %.4f'%(ece)
ax.add_line(line)
ax.set_xlabel('Logit Bins', fontsize=38, fontweight=1)
ax.set_ylabel('Accuracy (%)', fontsize=38, fontweight=1)
plt.legend(fontsize=13, loc= "best")
#plt.title(title_name, fontsize=32)
plt.title('MINT VGG16', fontsize=38)
#plt.title('Orig. VGG16', fontsize=38)
plt.tick_params(labelsize=30)
plt.setp(ax.spines.values(), linewidth=1, color ='k')
plt.xlim((0,1))
plt.ylim((0,1))
#fig.savefig('./calibration_orig_vgg16.pdf', format='pdf', dpi=1200)
fig.savefig('./calibration_mint_vgg16.pdf', format='pdf', dpi=1200)
#plt.show()
