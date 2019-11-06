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
 
from models                    import BasicBlock
from models                    import Alexnet       as alex
from models                    import Alexnet_mod2  as alex_mod2
from models                    import ResNet        as resnet 
from models                    import MLP           as mlp 

torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

def gen_mask(I_parent_file, prune_percent, parent_key, children_key, clusters, clusters_children, Labels_file, Labels_children_file, final_weights):
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

        sorted_weights = np.sort(sorted_weights)
        cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
        cutoff_value   = sorted_weights[cutoff_index]
        #print('Cutoff index %d wrt total number of elements %d' %(cutoff_index, sorted_weights.shape[0])) 
        #print('Cutoff value %f' %(cutoff_value)) 


        for num_layers in range(len(parent_key)):
            parent_k   = parent_key[num_layers]
            children_k = children_key[num_layers]

            for child in range(clusters_children[num_layers]):
                for group_1 in range(clusters[num_layers]):
                    if I_parent[str(num_layers)][child, group_1] <= cutoff_value:
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
                total_count += init_weights[children_key[num_layers]].reshape(-1).shape[0]
                valid_count += len(np.where(init_weights[children_key[num_layers]].reshape(-1)!=0.)[0])
            

        else:
            valid_count = len(np.where(init_weights[children_key[0]].reshape(-1)!= 0.0)[0])
            total_count = float(init_weights[children_key[0]].reshape(-1).shape[0])



        true_prune_percent = valid_count / float(total_count) * 100.

        ### Save Mask
        #np.save('logits_29_'+str(prune_percent*10)+'.npy', mask_weights)

 
        return mask_weights, true_prune_percent


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

def train(Epoch, Batch_size, Lr, Save_dir, Dataset, Dims, Milestones, Rerun, Opt, Weight_decay, Model, Gamma, Nesterov, Device_ids, Retrain, Retrain_mask, Labels_file, Labels_children_file, prune_percent):

    #print("Experimental Setup: ", args)

    np.random.seed(1993)
    total_acc = []


    # Tensorboard Element
    writer = SummaryWriter() 

    # Load Data
    trainloader, testloader = data_loader(Dataset, Batch_size)

   
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    if Model == 'alexnet':
        model = alex(num_classes=Dims).to(device)

    elif Model == 'alexnet2':
        model = alex_mod2(num_classes=Dims).to(device)

    elif Model == 'resnet':
        model = resnet(BasicBlock, [2,2,2,2], num_classes=Dims).to(device)

    elif Model == 'mlp':
        model = mlp(num_classes=Dims).to(device)

    else:
        print('Invalid optimizer selected. Exiting')
        exit(1)

    # END IF

    # Retrain option
    if Retrain:
        model.load_state_dict(load_checkpoint(Retrain))

        mask, true_prune_percent = gen_mask(Retrain_mask, prune_percent, ['fc1.weight','fc2.weight'], ['fc2.weight','fc3.weight'], [10, 10], [10, 10], Labels_file, Labels_children_file, load_checkpoint(Retrain))

        model.setup_masks(mask)

    logsoftmax = nn.LogSoftmax()

    # Prune-Loop
    params     = [p for p in model.parameters() if p.requires_grad]
    optimizer  = optim.SGD(params, lr=Lr, momentum=0.9, weight_decay=Weight_decay, nesterov=Nesterov)
    #optimizer  = optim.RMSprop(model.parameters(), lr=Lr)
    scheduler  = MultiStepLR(optimizer, milestones=Milestones, gamma=Gamma)    


    best_model_acc = 0.0

    # Training Loop
    for epoch in range(Epoch):
        running_loss = 0.0
        if not Retrain:
            print('Epoch: ', epoch)

        # Save Current Model
        save_checkpoint(epoch, 0, model, optimizer, Save_dir+'/'+str(0)+'/logits_'+str(epoch)+'.pkl')

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


            if x_input.shape[0] and x_input.shape[0] >= len(Device_ids):
                x_input, y_label = x_input.to(device), y_label.to(device)

                optimizer.zero_grad()

                outputs = model(x_input)
                loss    = torch.mean(torch.sum(-y_label * logsoftmax(outputs), dim=1))

                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()
            
                ## Add Loss Element
                writer.add_scalar(Dataset+'/'+Model+'/loss', loss.item(), epoch*len(trainloader) + step)
                if np.isnan(running_loss):
                    import pdb; pdb.set_trace()

                # END IF

            # END IF

            ########################### Data Loader + Training ##################################
 
            if step % 100 == 0 and not(Retrain):
                print('Epoch: ', epoch, '| train loss: %.4f' % (running_loss/100.))
                running_loss = 0.0

            # END IF
   
        scheduler.step()

        end_time = time.time()
        if not Retrain:
            print("Time for epoch: %f", end_time - start_time)
 
        epoch_acc = 100*accuracy(model, testloader, device)
        writer.add_scalar(Dataset+'/'+Model+'/accuracy', epoch_acc, epoch)

        if not Retrain:
            print('Accuracy of the network on the 10000 test images: %f %%\n' % (epoch_acc))


        if best_model_acc < epoch_acc:
            best_model_acc = epoch_acc
            save_checkpoint(epoch + 1, 0, model, optimizer, Save_dir+'/'+str(0)+'/logits_best.pkl')
    
    # END FOR

    # Close Tensorboard Element
    writer.close()

    # Save Final Model
    save_checkpoint(epoch + 1, 0, model, optimizer, Save_dir+'/'+str(0)+'/logits_final.pkl')
    total_acc.append(100.*accuracy(model, testloader, device))

    print('Highest accuracy obtained for pruning percentage %f is %f\n'%(true_prune_percent, best_model_acc))
        
    return total_acc 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--Epoch',                type=int   ,   default=10)
    parser.add_argument('--Batch_size',           type=int   ,   default=128)
    parser.add_argument('--Lr',                   type=float ,   default=0.001)
    parser.add_argument('--Save_dir',             type=str   ,   default='.')
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
    
    args = parser.parse_args()
 
    for prune_percent in np.arange(0.7, 0.8, step=0.01):
        acc = train(args.Epoch, args.Batch_size, args.Lr, args.Save_dir, args.Dataset, args.Dims, args.Milestones, args.Expt_rerun, args.Opt, args.Weight_decay, args.Model, args.Gamma, args.Nesterov, args.Device_ids, args.Retrain, args.Retrain_mask, args.Labels_file, args.Labels_children_file, prune_percent)
    
    #print('Average accuracy: ', np.mean(acc))
    #print('Peak accuracy: ',    np.max(acc))
    #print('Std. of accuracy: ', np.std(acc))
