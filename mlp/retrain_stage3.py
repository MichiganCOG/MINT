"""
Code Acknowledgements: https://morvanzhou.github.io/tutorials/
"""

import os
import cv2
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
import matplotlib.pyplot as plt

from utils                     import save_checkpoint, load_checkpoint, accuracy
from matplotlib                import cm
from torchvision               import datasets, transforms
from data_handler              import data_loader
from tensorboardX              import SummaryWriter
from torch.autograd            import Variable
from mpl_toolkits.mplot3d      import Axes3D
from torch.optim.lr_scheduler  import MultiStepLR
 
from model                     import MLP           as mlp 

#### Function to Generate Mask Using CMI Values ####
def gen_mask(I_parent_file, prune_percent, parent_key, children_key, clusters, clusters_children, Labels_file, Labels_children_file, final_weights, upper_prune_limit):
        I_parent        = np.load('results/'+I_parent_file, allow_pickle=True).item()
        labels          = np.load('results/'+Labels_file, allow_pickle=True).item()
        labels_children = np.load('results/'+Labels_children_file, allow_pickle=True).item()

        # Create a copy of trained weights
        init_weights   = copy.deepcopy(final_weights)

        sorted_weights = None
        mask_weights   = {}

        # Flatten I_parent dictionary
        for looper_idx in range(len(I_parent.keys())):
            if sorted_weights is None:
                sorted_weights = I_parent[str(looper_idx)].reshape(-1)
            else:
                sorted_weights =  np.concatenate((sorted_weights, I_parent[str(looper_idx)].reshape(-1)))

            # END IF

        # END FOR

        # Compute unique values
        sorted_weights = np.unique(sorted_weights)
        sorted_weights = np.sort(sorted_weights)
        cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
        cutoff_value   = sorted_weights[cutoff_index]

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

                            # END FOR

                        # END FOR

                    # END IF

                # END FOR

            # END FOR

            mask_weights[children_k] = np.ones(init_weights[children_k].shape)
            mask_weights[children_k][np.where(init_weights[children_k].detach().cpu()==0)] = 0

        # END FOR

        if len(parent_key) > 1:
            total_count = 0
            valid_count = 0

            for num_layers in range(len(parent_key)):
                total_count += init_weights[children_key[num_layers]].reshape(-1).shape[0]
                valid_count += len(np.where(init_weights[children_key[num_layers]].detach().cpu().reshape(-1)!=0.)[0])

            # END FOR

        else:
            valid_count = len(np.where(init_weights[children_key[0]].detach().cpu().reshape(-1)!= 0.0)[0])
            total_count = float(init_weights[children_key[0]].reshape(-1).shape[0])

        # END IF

        true_prune_percent = valid_count / float(total_count) * 100.

        return mask_weights, true_prune_percent, total_count


####  Function to Re-train DNN After Masking Weights ####
def train(Epoch, Batch_size, Lr, Dataset, Dims, Milestones, Rerun, Opt, Weight_decay, Model, Gamma, Nesterov, Device_ids, Retrain, Retrain_mask, Labels_file, Labels_children_file, prune_percent, parent_key, children_key, parent_clusters, children_clusters, upper_prune_limit):

    print("Experimental Setup: ", args)

    total_acc = []

    # Load Data
    trainloader, testloader, extraloader = data_loader(Dataset, Batch_size)

   
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    if Model == 'mlp':
        model = mlp(num_classes=Dims).to(device)

    else:
        print('Invalid optimizer selected. Exiting')
        exit(1)

    # END IF

    ## Retrain Setup ##

    # Load old state
    model.load_state_dict(load_checkpoint(Retrain))

    # Obtain masks
    mask, true_prune_percent, total_count = gen_mask(Retrain_mask, prune_percent, parent_key, children_key, parent_clusters, children_clusters, Labels_file, Labels_children_file, load_checkpoint(Retrain), upper_prune_limit)

    # Apply masks
    model.setup_masks(mask)

    ## Retrain Setup ##

    logsoftmax = nn.LogSoftmax()
    params     = [p for p in model.parameters() if p.requires_grad]

    if Opt == 'rms':
        optimizer  = optim.RMSprop(model.parameters(), lr=Lr)

    else:
        optimizer  = optim.SGD(params, lr=Lr, momentum=0.9, weight_decay=Weight_decay, nesterov=Nesterov)

    # END IF

    scheduler      = MultiStepLR(optimizer, milestones=Milestones, gamma=Gamma)    
    best_model_acc = 0.0
    best_model     = None

    # Training Loop
    for epoch in range(Epoch):
        running_loss = 0.0

        # Setup Model To Train 
        model.train()

        start_time = time.time()

        for step, data in enumerate(trainloader):
    
            # Extract Data From Loader
            x_input, y_label = data

            ########################### Data Loader + Training ##################################
            one_hot                                       = np.zeros((y_label.shape[0], Dims))
            one_hot[np.arange(y_label.shape[0]), y_label] = 1
            y_label                                       = torch.Tensor(one_hot) 


            if x_input.shape[0]:
                x_input, y_label = x_input.to(device), y_label.to(device)

                optimizer.zero_grad()

                outputs = model(x_input)
                loss    = torch.mean(torch.sum(-y_label * logsoftmax(outputs), dim=1))

                loss.backward()
                optimizer.step()
    
                ## Add Loss Element
                if np.isnan(loss.item()):
                    import pdb; pdb.set_trace()

                # END IF

            # END IF

            ########################### Data Loader + Training ##################################
 
        # END FOR
   
        scheduler.step()

        end_time = time.time()
 
        epoch_acc = 100*accuracy(model, testloader, device)


        if best_model_acc < epoch_acc:
            best_model_acc = epoch_acc
            best_model     = copy.deepcopy(model)

        # END IF
    
    # END FOR

    print('Requested prune percentage is %f'%(prune_percent))
    print('Highest accuracy for true pruning percentage %f is %f'%(true_prune_percent, best_model_acc))
    print('Total number of parameters is %d\n'%(total_count))

    return true_prune_percent, best_model_acc, best_model, optimizer  

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
    parser.add_argument('--Labels_children_file', type=str)
    parser.add_argument('--parent_key',           nargs='+',     type=str,       default=['conv1.weight'])
    parser.add_argument('--children_key',         nargs='+',     type=str,       default=['conv2.weight'])
    parser.add_argument('--parent_clusters',      nargs='+',     type=int,       default=[8])
    parser.add_argument('--children_clusters',    nargs='+',     type=int,       default=[8])
    parser.add_argument('--upper_prune_limit',    type=float,    default=0.75)
    parser.add_argument('--upper_prune_per',      type=float,    default=0.1)
    parser.add_argument('--lower_prune_per',      type=float,    default=0.9)
    parser.add_argument('--prune_per_step',       type=float,    default=0.001)
    
    # Keywords to save best re-trained file
    parser.add_argument('--Save_dir',             type=str   ,   default='.')

    parser.add_argument('--key_id',               type=int)
    
    args = parser.parse_args()
 
    possible_prune_percents   = np.arange(args.lower_prune_per, args.upper_prune_per, step=args.prune_per_step)

    true_prune_percent, best_model_acc, model, optimizer = train(args.Epoch, args.Batch_size, args.Lr, args.Dataset, args.Dims, args.Milestones, args.Expt_rerun, args.Opt, args.Weight_decay, args.Model, args.Gamma, args.Nesterov, args.Device_ids, args.Retrain, args.Retrain_mask, args.Labels_file, args.Labels_children_file, possible_prune_percents[args.key_id-1], args.parent_key, args.children_key, args.parent_clusters, args.children_clusters, args.upper_prune_limit)

    print('Saving best model: True prune percent %f, Best Acc. %f'%(true_prune_percent, best_model_acc))
    save_checkpoint(args.Epoch, 0, model, optimizer, args.Save_dir+'/0/logits_'+str(true_prune_percent)+'.pkl')
            
        
