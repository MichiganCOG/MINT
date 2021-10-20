""" Simple Code To Load MNIST Dataset """

import torch
import torchvision

import numpy             as np
import torch.utils.data  as Data

from torchvision  import datasets, transforms

####  Function to Load MNIST ####
def data_loader(dataset='MNIST', Batch_size = 64):


    if dataset == 'MNIST':

        train_data = datasets.MNIST('./data', 
                                    train=True, 
                                    download=True, 
                                    transform=transforms.ToTensor())
        
        test_data  = datasets.MNIST('./data', 
                                    train=False,
                                    download=True, 
                                    transform=transforms.ToTensor())
        
        extra_data = datasets.MNIST("./data", train=True,  transform=transforms.ToTensor(), download=False)
        

    else:
        print('Dataset selected isn\'t supported! Error.')
        exit(0)

    # END IF

    trainloader = torch.utils.data.DataLoader(dataset = train_data, batch_size=Batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset = test_data,  batch_size=Batch_size, shuffle=False, num_workers=6, pin_memory=True)
    extraloader = torch.utils.data.DataLoader(dataset = extra_data, batch_size=Batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader, extraloader
