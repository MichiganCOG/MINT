import torch
import copy
import torch

import numpy             as np
import matplotlib.pyplot as plt

from data_handler import data_loader
from models       import Alexnet        as alex
from utils        import save_checkpoint, load_checkpoint, accuracy


def calc_perf(prune_percent):
    init_weights   = load_checkpoint('/z/home/madantrg/Pruning/results/CIFAR10_ALEXNET_BATCH/0/logits_final.pkl')
    final_weights  = init_weights.copy()
    sorted_weights = None
    cutoff_value   = -100.0
    
    for item in init_weights.keys():
        if 'linear2.weight' in item or 'linear3.weight' in item:
            if sorted_weights is None:
                sorted_weights = np.abs(init_weights[item].reshape(-1))
            else:
                sorted_weights = np.hstack((sorted_weights, np.abs(init_weights[item].reshape(-1))))
    
            # END IF
    
    sorted_weights = np.sort(sorted_weights[np.where(sorted_weights!=0.)[0]])
    cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
    cutoff_value   = sorted_weights[cutoff_index]
    print('Cutoff index %d wrt total number of elements %d' %(cutoff_index, sorted_weights.shape[0])) 


    total_count = 0
    valid_count = 0
 
    for item in init_weights.keys():
        if 'linear2.weight' in item or 'linear3.weight' in item:
            orig_shape          = final_weights[item].shape
            sorted_weights      = np.abs(final_weights[item].reshape(-1)).numpy()
            cutoff_indices      = np.where(sorted_weights < cutoff_value)[0]        
            final_weights[item] = final_weights[item].reshape(-1)
            final_weights[item][cutoff_indices] = 0.0
            final_weights[item] = final_weights[item].reshape(orig_shape)

            # Verification
            total_count += final_weights[item].reshape(-1).shape[0]
            valid_count += len(np.where(final_weights[item].reshape(-1)!=0.)[0])
 
    print('Percent weights remaining from 2 layers is %f'%(valid_count/float(total_count)*100.))
    true_prune_percent = valid_count/float(total_count)*100.

    # Load Data
    trainloader, testloader = data_loader('CIFAR10', 128)
    
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    model = alex(num_classes=10).to(device)
    model.load_state_dict(final_weights)
    
    torch.save(model.state_dict(), 'alexnet_stl10_'+str(int(prune_percent*10))+'.pkl')
 
    acc = 100.*accuracy(model, testloader, device) 
    print('Accuracy of the pruned network on the 10000 test images: %f %%\n' %(acc))

    return acc, true_prune_percent


if __name__=='__main__':
    print "Calculation of performance change for one-shot pruning based on weights"

    perf      = None
    prune_per = None

    for prune_percent in np.arange(0.0, 1.0, step=0.1):
        acc, true_prune_percent = calc_perf(prune_percent)
        if perf is None:
            perf      = [acc]
            prune_per = [true_prune_percent]

        else:
            perf.append(acc)
            prune_per.append(true_prune_percent)

        # END IF

    plt.plot(prune_per, perf) 
    plt.xlabel('Ratio of weights pruned')
    plt.ylabel('Performance of AlexNet')
    plt.title('Comparison of performance variation when AlexNet is one-shot pruned, w/o retraining')
    plt.show()
