import torch

import numpy    as np
import torch.nn as nn

#### Compute And Return Accuracy ####
def accuracy(net, testloader, device):
    correct = 0
    total   = 0
    net.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs      = net(images, labels=True)
            _, predicted = torch.max(outputs.data, 1)

            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

        # END FOR

    # END WITH

    return float(correct) / total
