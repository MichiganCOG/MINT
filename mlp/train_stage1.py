"""
Code Acknowledgements: https://morvanzhou.github.io/tutorials/
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
 
from model                     import MLP           as mlp 

torch.backends.cudnn.deterministic = True
random.seed(999)
np.random.seed(999)
torch.manual_seed(999)

#### Function to Train DNN ####
def train(Epoch, Batch_size, Lr, Save_dir, Dataset, Dims, Milestones, Rerun, Opt, Weight_decay, Model, Gamma, Nesterov, Device_ids):

    print("Experimental Setup: ", args)

    np.random.seed(1993)

    total_acc = []

    for total_iteration in range(Rerun):

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

        logsoftmax = nn.LogSoftmax()
        params     = [p for p in model.parameters() if p.requires_grad]

        if Opt == 'rms':
            optimizer  = optim.RMSprop(model.parameters(), lr=Lr)

        else:
            optimizer  = optim.SGD(params, lr=Lr, momentum=0.9, weight_decay=Weight_decay, nesterov=Nesterov)

        # END IF

        scheduler      = MultiStepLR(optimizer, milestones=Milestones, gamma=Gamma)    
        best_model_acc = 0.0

        # Training Loop
        for epoch in range(Epoch):
            running_loss = 0.0
            print('Epoch: ', epoch)

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
        
                    running_loss += loss.item()
                
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

            print('Accuracy of the network on the 10000 test images: %f %%\n' % (epoch_acc))

            if best_model_acc < epoch_acc:
                best_model_acc = epoch_acc
                save_checkpoint(epoch + 1, 0, model, optimizer, Save_dir+'/'+str(total_iteration)+'/logits_best.pkl')

            # END IF
        
        # END FOR

        # Save Final Model
        save_checkpoint(epoch + 1, 0, model, optimizer, Save_dir+'/'+str(total_iteration)+'/logits_final.pkl')
        total_acc.append(100.*accuracy(model, testloader, device))

        print('Highest accuracy obtained is %f'%(best_model_acc))
        

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
    
    args = parser.parse_args()
 
    train(args.Epoch, args.Batch_size, args.Lr, args.Save_dir, args.Dataset, args.Dims, args.Milestones, args.Expt_rerun, args.Opt, args.Weight_decay, args.Model, args.Gamma, args.Nesterov, args.Device_ids)
