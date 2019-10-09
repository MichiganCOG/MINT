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
 
from models                    import Alexnet        as alex

torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())

    not_zero_idxs = np.where(np.array(all_weights)!=0.0)[0]
    threshold = np.percentile(np.array(all_weights)[not_zero_idxs], pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            #pruned_inds = (p.data.abs() != 0) & (p.data.abs() > threshold)
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks

def pruned_weights(model, prune_percent=0.6):
    init_weights    = model.state_dict()
    final_weights   = init_weights.copy()
    sorted_weights  = None
    cutoff_value    = -100.0
    
    for item in init_weights.keys():
        #if 'weight' in item:
        if sorted_weights is None:
            sorted_weights = np.abs(init_weights[item].reshape(-1))
        else:
            sorted_weights = np.hstack((sorted_weights, np.abs(init_weights[item].reshape(-1))))
    
        #    # END IF
    
    sorted_weights = np.sort(sorted_weights)
    not_zero_idxs  = np.where(sorted_weights!=0.0)[0]
    cutoff_index   = np.round(prune_percent * sorted_weights[not_zero_idxs].shape[0]).astype('int')
    cutoff_value   = sorted_weights[not_zero_idxs][cutoff_index]
    print('Cutoff value: ', cutoff_value)
    
    for item in init_weights.keys():
        #if 'weight' in item:
        orig_shape          = final_weights[item].shape
        sorted_weights      = np.abs(final_weights[item].reshape(-1)).numpy()
        cutoff_indices      = np.where(sorted_weights < cutoff_value)[0]        
        final_weights[item] = final_weights[item].reshape(-1)
        final_weights[item][cutoff_indices] = 0.0
        print('Number of zeroed weights is %d'%(len(np.where(final_weights[item]==0.0)[0])))
        final_weights[item] = final_weights[item].reshape(orig_shape)
    

    model.load_state_dict(final_weights)

    return model

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

def train(Epoch, Batch_size, Lr, Save_dir, Dataset, Dims, Milestones, Rerun, Opt, Weight_decay, Model, Gamma, Nesterov, Device_ids, Prune_steps):

    print("Experimental Setup: ", args)

    np.random.seed(1993)
    total_acc = []

    for total_iteration in range(Rerun):

        # Tensorboard Element
        writer = SummaryWriter() 

        # Load Data
        trainloader, testloader = data_loader(Dataset, Batch_size)

   
        # Check if GPU is available (CUDA)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load Network
        if Model == 'alexnet':
            model = alex(num_classes=Dims).to(device)

        else:
            print('Invalid optimizer selected. Exiting')
            exit(1)

        # END IF

        logsoftmax = nn.LogSoftmax()

        # Prune-Loop
        for prune_loop in range(Prune_steps):
            params     = [p for p in model.parameters() if p.requires_grad]
            optimizer  = optim.SGD(params, lr=Lr, momentum=0.9, weight_decay=Weight_decay, nesterov=Nesterov)
            scheduler  = MultiStepLR(optimizer, milestones=Milestones, gamma=Gamma)    

            # Training Loop
            for epoch in range(Epoch):
                running_loss = 0.0
                print('Epoch: ', epoch)

                # Save Current Model
                save_checkpoint(epoch, 0, model, optimizer, Save_dir+'/'+str(total_iteration)+'/logits_'+str(epoch)+'.pkl')

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
 
                    if step % 100 == 0:
                        print('Epoch: ', epoch, '| train loss: %.4f' % (running_loss/100.))
                        running_loss = 0.0

                    # END IF
   
                # END FOR
                scheduler.step()

                end_time = time.time()
                print("Time for epoch: %f", end_time - start_time)
 
                epoch_acc = 100*accuracy(model, testloader, device)
                writer.add_scalar(Dataset+'/'+Model+'/accuracy', epoch_acc, epoch)

                print('Accuracy of the network on the 10000 test images: %f %%\n' % (epoch_acc))
            
            # END FOR
            if prune_loop != Prune_steps-1:
                masks = weight_prune(model, 60)
                model.set_masks(masks) 

        # END FOR

        # Close Tensorboard Element
        writer.close()

        # Save Final Model
        save_checkpoint(epoch + 1, 0, model, optimizer, Save_dir+'/'+str(total_iteration)+'/logits_final.pkl')
        total_acc.append(100.*accuracy(model, testloader, device))
        
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
    parser.add_argument('--Weight_decay',         type=float ,   default=0.001)
    parser.add_argument('--Model',                type=str   ,   default='resnet32')
    parser.add_argument('--Gamma',                type=float ,   default=0.1)
    parser.add_argument('--Nesterov',             action='store_true' , default=False)
    parser.add_argument('--Device_ids',           nargs='+',     type=int,       default=[0])
    parser.add_argument('--Prune_steps',          type=int   ,   default=2)
    
    args = parser.parse_args()
 
    acc = train(args.Epoch, args.Batch_size, args.Lr, args.Save_dir, args.Dataset, args.Dims, args.Milestones, args.Expt_rerun, args.Opt, args.Weight_decay, args.Model, args.Gamma, args.Nesterov, args.Device_ids, args.Prune_steps)
    
    print('Average accuracy: ', np.mean(acc))
    print('Peak accuracy: ',    np.max(acc))
    print('Std. of accuracy: ', np.std(acc))
