import torch
import copy

import numpy             as np
import matplotlib.pyplot as plt

from data_handler            import data_loader
from models                  import Alexnet        as alex
from utils                   import save_checkpoint, load_checkpoint, accuracy
from knn                     import *
from tqdm                    import tqdm
from scipy.cluster.vq        import vq, kmeans2, whiten
from sklearn.decomposition   import PCA
from scipy.cluster.hierarchy import fclusterdata

def pruner(prune_percent, parent_key, children_key, children, final_weights, model, testloader, device, ret=False):

    # Create a copy
    init_weights   = copy.deepcopy(final_weights)

    sorted_weights = None

    # Flatten I_parent dictionary
    for looper_idx in [len(parent_key)-1]:
        if sorted_weights is None:
            sorted_weights = init_weights[children_key[looper_idx]].reshape(-1)
        else:
            sorted_weights =  np.concatenate((sorted_weights, init_weights[children_key[looper_idx]].reshape(-1)))

    sorted_weights = np.sort(np.abs(sorted_weights))
    cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
    cutoff_value   = sorted_weights[cutoff_index]
    print('Cutoff index %d wrt total number of elements %d' %(cutoff_index, sorted_weights.shape[0])) 
    print('Cutoff value %f' %(cutoff_value)) 



    for num_layers in [len(parent_key)-1]:
        children_k = children_key[num_layers]
        init_weights[children_k][np.where(np.abs(init_weights[children_k].cpu().numpy()) < cutoff_value)] = 0.0

    # END FOR

    if len(parent_key) > 1:
        total_count = 0
        valid_count = 0
        for num_layers in range(len(parent_key)):
            total_count += init_weights[children_key[num_layers]].reshape(-1).shape[0]
            valid_count += len(np.where(init_weights[children_key[num_layers]].reshape(-1)!=0.)[0])
    
        if ret:    
            print('Percent weights remaining from %d layers is %f'%(len(parent_key), valid_count/float(total_count)*100.))

    else:
        valid_count = len(np.where(init_weights[children_key[0]].reshape(-1)!= 0.0)[0])
        total_count = float(init_weights[children_key[0]].reshape(-1).shape[0])
   
        if ret:
            print('Percent weights remaining from %d layers is %f'%(len(parent_key), valid_count/total_count*100.))



    true_prune_percent = valid_count / float(total_count) * 100.

    model.load_state_dict(init_weights)
    acc = 100.*accuracy(model, testloader, device) 
    print('Accuracy of the pruned network on the 10000 test images: %f %%\n' %(acc))
   
    if ret:
        return acc, true_prune_percent, init_weights

    else:
        return acc, true_prune_percent

    # END IF

def calc_perf(parent_key, children_key, alg, clusters=2):
    init_weights   = load_checkpoint('/z/home/madantrg/Pruning/results/STL10_ALEXNET_BATCH/0/logits_final.pkl')

    # Load Data
    trainloader, testloader = data_loader('STL10', 128)
 
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    model = alex(num_classes=10).to(device)
    model.load_state_dict(init_weights)
    model.eval()

    nlayers      = len(parent_key)
    labels       = {}
    children     = {}
    I_parent     = {}


    for num_layers in range(len(parent_key)):
        for idx in [num_layers]:
            children[str(idx)] = init_weights[children_key[idx]].shape[0]

        # END FOR        
 
        print("----------------------------------")
        print("Begin Pruning of Weights")

        perf         = None
        prune_per    = None

        for prune_percent in np.arange(0.0, 1.0, step=0.05):
            acc, true_prune_percent = pruner(prune_percent, parent_key[:num_layers+1], children_key[:num_layers+1], children, init_weights, model, testloader, device, False)
            if perf is None:
                perf      = [acc]
                prune_per = [true_prune_percent]

            else:
                perf.append(acc)
                prune_per.append(true_prune_percent)

            # END IF

        acc, true_prune_percent, init_weights = pruner(np.arange(0.0, 1.0, step=0.05)[np.where(np.array(perf) == perf[np.where(np.array(perf) >= perf[0])[0][-1]])[0][-1]], parent_key[:num_layers+1], children_key[:num_layers+1], children, init_weights, model, testloader, device, True)


    return perf, prune_per

if __name__=='__main__':
    print "Calculation of performance change for one-shot pruning based on Weight values"

    perf         = None
    prune_per    = None
    parent_key   = ['conv5.weight']#['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight',  'linear1.weight','linear2.weight']
    children_key = ['linear1.weight']#['conv2.weight','conv3.weight','conv4.weight','conv5.weight','linear1.weight','linear2.weight','linear3.weight']
    alg          = '1a_group'
    clusters     = 4 

    perf, prune_per = calc_perf(parent_key, children_key, alg, clusters)

    plt.plot(prune_per, perf) 
    plt.xlabel('Ratio of weights pruned')
    plt.ylabel('Performance of AlexNet')
    plt.title('Comparison of performance variation when AlexNet is one-shot pruned, w/o retraining')
    plt.show()
