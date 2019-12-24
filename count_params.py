"""
LEGACY:
    View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
    My Youtube Channel: https://www.youtube.com/user/MorvanZhou
    Dependencies:
    torch: 0.4
    matplotlib
    numpy
"""
import os
import time
import copy
import math
import torch
import random
import argparse
import torchvision
import torch.utils.data.distributed

import numpy             as np
import torch.nn          as nn
import torch.optim       as optim
import torch.utils.data  as Data

from utils                     import save_checkpoint, load_checkpoint, accuracy
from torchvision               import datasets, transforms
from data_handler              import data_loader
from torch.autograd            import Variable
from torch.optim.lr_scheduler  import MultiStepLR
 
from models                    import BasicBlock
from models                    import Alexnet       as alex
from models                    import Resnet56      as resnet 
from models                    import MLP           as mlp 
from models                    import VGG16_bn      as vgg 

torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

def gen_mask(I_parent_file, prune_percent, parent_key, children_key, clusters, clusters_children, Labels_file, Labels_children_file, final_weights, upper_prune_limit):
        I_parent        = np.load(I_parent_file).item()
        labels          = np.load(Labels_file).item()
        labels_children = np.load(Labels_children_file).item()

        # Create a copy
        init_weights   = copy.deepcopy(final_weights)

        sorted_weights = None
        mask_weights   = {}

        # Flatten I_parent dictionary
        for looper_idx in range(len(I_parent.keys())):
            if sorted_weights is None:
                sorted_weights = I_parent[str(looper_idx)].reshape(-1)
            else:
                sorted_weights =  np.concatenate((sorted_weights, I_parent[str(looper_idx)].reshape(-1)))


        # Compute unique values
        sorted_weights = np.unique(sorted_weights)

        sorted_weights = np.sort(sorted_weights)
        cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
        cutoff_value   = sorted_weights[cutoff_index]
        #print('Cutoff index %d wrt total number of elements %d' %(cutoff_index, sorted_weights.shape[0])) 
        #print('Cutoff value %f' %(cutoff_value)) 


        for num_layers in range(len(parent_key)):
            parent_k   = parent_key[num_layers]
            children_k = children_key[num_layers]

            for child in range(clusters_children[num_layers]):

                # Pre-compute % of weights to be removed in layer
                layer_remove_per = float(len(np.where(I_parent[str(num_layers)].reshape(-1) <= cutoff_value)[0]) * (init_weights[children_k].shape[0]/ clusters[num_layers])* (init_weights[children_k].shape[1]/clusters_children[num_layers])) / np.prod(init_weights[children_k].shape[:2])

                if layer_remove_per >= upper_prune_limit:
                    local_sorted_weights = np.sort(np.unique(I_parent[str(num_layers)].reshape(-1)))
                    cutoff_value_local   = local_sorted_weights[np.round(upper_prune_limit * local_sorted_weights.shape[0]).astype('int')]
                
                else:
                    cutoff_value_local = cutoff_value

                # END IF

                for group_1 in range(clusters[num_layers]):
                    if (I_parent[str(num_layers)][child, group_1] <= cutoff_value_local):
                        for group_p in np.where(labels[str(num_layers)]==group_1)[0]:
                            for group_c in np.where(labels_children[str(num_layers)]==child)[0]:
                                init_weights[children_k][group_c, group_p] = 0.

                    # END IF

                # END FOR

            # END FOR

            mask_weights[children_k] = np.ones(init_weights[children_k].shape)
            mask_weights[children_k][np.where(init_weights[children_k]==0)] = 0

        # END FOR

        if len(parent_key) > 1:
            total_count = 0
            valid_count = 0

            for num_layers in range(len(parent_key)):
                #total_count = 0
                #valid_count = 0
                total_count += init_weights[children_key[num_layers]].reshape(-1).shape[0]
                valid_count += len(np.where(init_weights[children_key[num_layers]].reshape(-1)!=0.)[0])
                #print('Compression percentage in layer %s is %f'%(children_key[num_layers], valid_count))

        else:
            valid_count = len(np.where(init_weights[children_key[0]].reshape(-1)!= 0.0)[0])
            total_count = float(init_weights[children_key[0]].reshape(-1).shape[0])



        true_prune_percent = valid_count / float(total_count) * 100.

        return mask_weights, true_prune_percent, total_count


def set_lr(optimizer, lr_update, utype='const'):
    for param_group in optimizer.param_groups:

        if utype == 'const':
            current_lr = param_group['lr']
            #print("Updating LR to ", lr_update)
            param_group['lr'] = lr_update

        else:
            current_lr = param_group['lr']
            print("Updating LR to ", current_lr*lr_update)
            param_group['lr'] = current_lr * lr_update
            current_lr*= lr_update

        # END IF

    # END FOR

    return optimizer

def train(Dims, Model, Device_ids):

    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    if Model == 'alexnet':
        model = alex(num_classes=Dims).to(device)

    elif Model == 'mlp':
        model = mlp(num_classes=Dims).to(device)

    elif Model == 'vgg':
        model = vgg(num_classes=Dims).to(device)

    elif Model == 'resnet':
        model = resnet(num_classes=Dims).to(device)

    else:
        print('Invalid optimizer selected. Exiting')
        exit(1)

    # END IF

    for name, layer in model._modules.items():
        if 'conv' in name or 'linear' in name:
            print('Number of parameters in layer %s is %d'%(name, model.state_dict()[name+'.weight'].reshape(-1).shape[0]))

        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--Epoch',                type=int   ,   default=10)
    parser.add_argument('--Batch_size',           type=int   ,   default=128)
    parser.add_argument('--Lr',                   type=float ,   default=0.001)
    parser.add_argument('--Dataset',              type=str   ,   default='CIFAR10')
    parser.add_argument('--Dims',                 type=int   ,   default=10)
    parser.add_argument('--Expt_rerun',           type=int   ,   default=1)
    parser.add_argument('--Milestones',           nargs='+',     type=float,       default=[100,150,200])
    parser.add_argument('--Opt',                  type=str   ,   default='sgd')
    parser.add_argument('--Weight_decay',         type=float ,   default=0.0001)
    parser.add_argument('--Model',                type=str   ,   default='resnet32')
    parser.add_argument('--Gamma',                type=float ,   default=0.1)
    parser.add_argument('--Nesterov',             action='store_true' , default=False)
    parser.add_argument('--Device_ids',           nargs='+',     type=int,       default=[0])
    parser.add_argument('--Retrain',              type=str)
    parser.add_argument('--Retrain_mask',         type=str)
    parser.add_argument('--Labels_file',          type=str)
    parser.add_argument('--Labels_children_file',          type=str)
    parser.add_argument('--parent_key',           nargs='+',     type=str,       default=['conv1.weight'])
    parser.add_argument('--children_key',         nargs='+',     type=str,       default=['conv2.weight'])
    parser.add_argument('--parent_clusters',      nargs='+',     type=int,       default=[8])
    parser.add_argument('--children_clusters',    nargs='+',     type=int,       default=[8])
    parser.add_argument('--upper_prune_limit',    type=float,    default=0.75)
    parser.add_argument('--upper_prune_per',      type=float,    default=0.1)
    parser.add_argument('--lower_prune_per',      type=float,    default=0.9)
    parser.add_argument('--prune_per_step',       type=float,    default=0.001)
    
    args = parser.parse_args()
 
    train(args.Dims, args.Model, args.Device_ids)
    
