import numpy as np

directory = '/z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH_PARALLEL/0/'

prefix = 'p_25g_'
load_keys = ['c1_c2', 'c2_c3', 'c3_c4', 'c4_c5', 'c5_c6', 'c6_c7', 'c7_c8', 'c8_c9', 'c9_c10', 'c10_c11', 'c11_c12', 'c12_c13', 'c13_l1', 'l1_l3']

res_I_dict               = {}
res_Labels_dict          = {}
res_Labels_children_dict = {}

for looper in range(len(load_keys)):
    res_I_dict[str(looper)]               = np.load(directory+'I_parent_'+prefix+load_keys[looper]+'.npy').item()['0']
    res_Labels_dict[str(looper)]          = np.load(directory+'Labels_'+prefix+load_keys[looper]+'.npy').item()['0']
    res_Labels_children_dict[str(looper)] = np.load(directory+'Labels_children_'+prefix+load_keys[looper]+'.npy').item()['0']

np.save(directory+'I_parents_p_25g.npy', res_I_dict)
np.save(directory+'Labels_p_25g.npy', res_Labels_dict)
np.save(directory+'Labels_children_p_25g.npy', res_Labels_children_dict)
