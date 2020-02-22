import torch
import torchvision

import numpy             as np
import torch.utils.data  as Data

from torchvision  import datasets, transforms

def data_loader(dataset='CIFAR10', Batch_size = 16, pre='cutout'):


    if dataset == 'IMAGENET':

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

        extra_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        test_transform  = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    
        train_data = datasets.ImageFolder('/z/dat/ImageNet_2012/train/', transform=train_transform)
        extra_data = datasets.ImageFolder('/z/dat/ImageNet_2012/val/',   transform=extra_transform)
        test_data  = datasets.ImageFolder('/z/dat/ImageNet_2012/val/',   transform=test_transform)

    else:
        print('Dataset selected isn\'t supported! Error.')
        exit(0)

    # END IF

    # EDIT TO TRUE for trainloader
    trainloader = torch.utils.data.DataLoader(dataset = train_data, batch_size=Batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset = test_data,  batch_size=Batch_size, shuffle=False, num_workers=6, pin_memory=True)
    extraloader = torch.utils.data.DataLoader(dataset = extra_data, batch_size=Batch_size, shuffle=False, num_workers=6, pin_memory=True)

    return trainloader, testloader, extraloader
