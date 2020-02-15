import torch
import numpy    as np
import torch.nn as nn

def accuracy(net, testloader, device):
    correct = 0
    total   = 0
    net.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            #Var = []
            #for looper in range(10):
            outputs      = net(images, labels=True)
            _, predicted = torch.max(outputs.data, 1)
                #if len(Var):
                #    Var = np.dstack((outputs.cpu().numpy(), Var))
                #else:
                #    Var = outputs.cpu().numpy()

            total   += labels.size(0)

            correct += (predicted == labels).sum().item()
    return float(correct) / total
