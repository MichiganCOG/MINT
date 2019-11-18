import torch
import torchvision

import numpy             as np
import torch.utils.data  as Data

from torchvision  import datasets, transforms
from cifarDataset import CIFAR10, CIFAR100

def data_loader(dataset='CIFAR10', Batch_size = 64, pre='cutout'):


    if dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize])

        
        extra_transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])
        
        train_data = datasets.CIFAR10("data", train=True,  transform=train_transform, download=False)
        test_data  = datasets.CIFAR10("data", train=False, transform=test_transform,  download=False)
        extra_data = datasets.CIFAR10("data", train=True,  transform=extra_transform, download=False)

    elif dataset == 'CIFAR100':

        mean = [0.5071, 0.4867, 0.4408]
        std  = [0.2675, 0.2275, 0.2761]

        train_transform = transforms.Compose([transforms.Resize((227,227)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        
        train_data = datasets.CIFAR100("data", train=True,  transform=train_transform, download=True)
        test_data  = datasets.CIFAR100("data", train=False, transform=test_transform,  download=True)
        
    elif dataset == 'STL10':

        train_transform = transforms.Compose([transforms.Resize((227,227)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        
        test_transform = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        train_data = datasets.STL10("data", split='train', transform=train_transform, download=False)
        test_data  = datasets.STL10("data", split='test',  transform=test_transform,  download=False)


    elif dataset == 'MNIST':

        train_data = datasets.MNIST('./data', 
                                       train=True, 
                                       download=True, 
                                       transform=transforms.ToTensor())
        
        test_data  = datasets.MNIST('./data', 
                                            train=False, 
                                            transform=transforms.ToTensor())
        
        extra_data = datasets.MNIST("./data", train=True,  transform=transforms.ToTensor(), download=False)
        

    else:
        print('Dataset selected isn\'t supported! Error.')
        exit(0)

    # END IF

    # EDIT TO TRUE for trainloader
    trainloader = torch.utils.data.DataLoader(dataset = train_data, batch_size=Batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset = test_data,  batch_size=Batch_size, shuffle=False, num_workers=2,pin_memory=True)
    extraloader = torch.utils.data.DataLoader(dataset = extra_data, batch_size=Batch_size, shuffle=False,  num_workers=2, pin_memory=True)

    return trainloader, testloader, extraloader
