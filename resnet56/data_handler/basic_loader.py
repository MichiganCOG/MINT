""" Simple Code To Load CIFAR10 Dataset """

import torch
import torchvision

import numpy             as np
import torch.utils.data  as Data

from torchvision  import datasets, transforms

def data_loader(dataset='MNIST', Batch_size = 64):


    if dataset == 'CIFAR10':
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])

        train_transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

        
        extra_transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        
        train_data = datasets.CIFAR10("data", train=True,  transform=train_transform, download=True)
        test_data  = datasets.CIFAR10("data", train=False, transform=test_transform,  download=True)
        extra_data = datasets.CIFAR10("data", train=True,  transform=extra_transform, download=True)


    else:
        print('Dataset selected isn\'t supported! Error.')
        exit(0)

    # END IF

    trainloader = torch.utils.data.DataLoader(dataset = train_data, batch_size=Batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset = test_data,  batch_size=Batch_size, shuffle=False, num_workers=6, pin_memory=True)
    extraloader = torch.utils.data.DataLoader(dataset = extra_data, batch_size=Batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader, extraloader
