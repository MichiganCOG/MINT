import torch
import copy

import numpy             as np
import matplotlib.pyplot as plt

from data_handler import data_loader
from models       import Alexnet        as alex
from utils        import save_checkpoint, load_checkpoint, accuracy
from knn          import *
from tqdm         import tqdm

def activations_2(data_loader, model, device):
    parents_op  = None
    children_op = None

    with torch.no_grad():
        # Collect linear1 outputs
        for step, data in enumerate(data_loader):
            # Extract Data From Loader
            x_input, y_label = data

            if parents_op is None:
                parents_op  = model(x_input.to(device), linear2=True).cpu().numpy()
                children_op = model(x_input.to(device), linear3=True).cpu().numpy()

            else:
                parents_op  = np.vstack((model(x_input.to(device), linear2=True).cpu().numpy(), parents_op))
                children_op = np.vstack((model(x_input.to(device), linear3=True).cpu().numpy(), children_op))

            # END IF 

        # END FOR

    # END FOR

    #np.save('parents_orig.npy', parents_op)
    #np.save('children_orig.npy', children_op)

    #parents_op  = np.load('parents_orig.npy')
    #children_op = np.load('children_orig.npy')

    return parents_op, children_op

def activations_1(data_loader, model, device):
    parents_op  = None
    children_op = None

    with torch.no_grad():
        # Collect linear1 outputs
        for step, data in enumerate(data_loader):
            # Extract Data From Loader
            x_input, y_label = data

            if parents_op is None:
                parents_op  = model(x_input.to(device), linear1=True).cpu().numpy()
                children_op = model(x_input.to(device), linear2=True).cpu().numpy()

            else:
                parents_op  = np.vstack((model(x_input.to(device), linear1=True).cpu().numpy(), parents_op))
                children_op = np.vstack((model(x_input.to(device), linear2=True).cpu().numpy(), children_op))

            # END IF 

        # END FOR

    # END FOR

    #np.save('parents_orig.npy', parents_op)
    #np.save('children_orig.npy', children_op)

    #parents_op  = np.load('parents_orig.npy')
    #children_op = np.load('children_orig.npy')

    return parents_op, children_op

def calc_perf(prune_percent):
    init_weights   = load_checkpoint('/z/home/madantrg/Pruning/results/STL10_ALEXNET_BATCH/0/logits_final.pkl')

    # Load Data
    trainloader, testloader = data_loader('STL10', 128)
 
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    model = alex(num_classes=10).to(device)
    model.load_state_dict(init_weights)
    model.eval()

    p1_op, c1_op = activations_1(trainloader, model, device)
    p2_op, c2_op = activations_2(trainloader, model, device)


    children     = [8, 10]
    I_parent     = np.zeros((2, 10, p1_op.shape[1]))
    #exclude_list = []
    #delta        = [0.105506, 0.172271, 0.279480, 0.760561, 2.352821, 2.880533, 10.425170, 15.637444, 17.923638, 24.074013]



    #for overall in range(2):
    #    for child in tqdm(range(children[overall])):
    #        for parent_1 in range(p1_op.shape[1]):
    #            for parent_2 in range(p1_op.shape[1]):
    #                if parent_1 == parent_2:
    #                    continue

    #                if overall == 0:
    #                    I_parent[overall, child, parent_1] += knn_mi(c1_op[:, child], p1_op[:, parent_1], np.around(p1_op[:, parent_2], decimals=1)) 

    #                else:
    #                    I_parent[overall, child, parent_1] += knn_mi(c2_op[:, child], p2_op[:, parent_1], np.around(p2_op[:, parent_2], decimals=1)) 

    #            # END FOR
 
    #        # END FOR 

    #    # END FOR

    #np.save('i_parent.npy', I_parent)
    I_parent = np.load('i_parent.npy')

    sorted_weights = np.sort(np.concatenate((I_parent[0][:8].reshape(-1), I_parent[1].reshape(-1))))
    cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
    cutoff_value   = sorted_weights[cutoff_index]
    print('Cutoff index %d wrt total number of elements %d' %(cutoff_index, sorted_weights.shape[0])) 
    print('Cutoff value %f' %(cutoff_value)) 



    parent_key   = 'linear1.weight'
    children_key = 'linear2.weight'
    for child in range(children[0]):
        for parent_1 in range(p1_op.shape[1]):
            if I_parent[0][child][parent_1] < cutoff_value:
                init_weights[children_key][child, parent_1] = 0.

             # END IF

    parent_key   = 'linear2.weight'
    children_key = 'linear3.weight'
    for child in range(children[1]):
        for parent_1 in range(p1_op.shape[1]):
            if I_parent[1][child][parent_1] < cutoff_value:
                init_weights[children_key][child, parent_1] = 0.

             # END IF

    model.load_state_dict(init_weights)
    acc = 100.*accuracy(model, testloader, device) 
    print('Accuracy of the pruned network on the 10000 test images: %f %%\n' %(acc))

    return acc

if __name__=='__main__':
    print "Calculation of performance change for one-shot pruning based on Mutual Information"

    perf = None

    for prune_percent in np.arange(0.0, 1.0, step=0.1):
        if perf is None:
            perf = [calc_perf(prune_percent)]

        else:
            perf.append(calc_perf(prune_percent))

        # END IF

    plt.plot(np.arange(0.0, 1.0, step=0.1), perf) 
    plt.xlabel('Ratio of weights pruned')
    plt.ylabel('Performance of AlexNet')
    plt.title('Comparison of performance variation when AlexNet is one-shot pruned, w/o retraining')
    plt.show()
