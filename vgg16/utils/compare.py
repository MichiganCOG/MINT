import os
import sys
import torch

import numpy as np

def compare(orig_file, compressed_file):

    assert(len(orig_file.keys()) == len(compressed_file.keys()))

    total_params = 0
    exist_params = 0

    for key in orig_file.keys():

        if 'bn' in key or 'bias' in key:
            continue
        non_zero_params = len(np.where(compressed_file[key].reshape(-1).cpu()!=0)[0])

        print('Percentage of parameters removed in layer %s is %f'%(key, non_zero_params/float(orig_file[key].reshape(-1).cpu().shape[0])))
        exist_params += non_zero_params 
        total_params += orig_file[key].reshape(-1).shape[0]

    import pdb; pdb.set_trace()
    print('Final compression percentage for VGG16 on CIFAR10 is %f'%(1 - (exist_params/float(total_params))))


if __name__=="__main__":

    orig_file       = torch.load('/z/home/madantrg/Pruning/vgg16/results/BASELINE_CIFAR10_VGG16_BN/0/logits_best.pkl')['state_dict']

    for compressed_file in ['/z/home/madantrg/Pruning/vgg16/results/BASELINE_CIFAR10_VGG16_BN_RETRAIN_1/0/logits_19.203483150080427.pkl']:
        compressed_file = torch.load(compressed_file)['state_dict']

        compare(orig_file, compressed_file)
