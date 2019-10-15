import torch
import copy

import numpy             as np
import matplotlib.pyplot as plt

from data_handler import data_loader
from models       import Alexnet        as alex
from utils        import save_checkpoint, load_checkpoint, accuracy
from knn          import *
from tqdm         import tqdm

def activations(data_loader, model, device, parent_linear1, parent_linear2, parent_linear3, child_linear1, child_linear2, child_linear3):
    parents_op  = None
    children_op = None

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data

            if parents_op is None:
                parents_op  = model(x_input.to(device), linear1=parent_linear1, linear2=parent_linear2, linear3=parent_linear3).cpu().numpy()
                children_op = model(x_input.to(device), linear1=child_linear1, linear2=child_linear2, linear3=child_linear3).cpu().numpy()

            else:
                parents_op  = np.vstack((model(x_input.to(device), linear1=parent_linear1, linear2=parent_linear2, linear3=parent_linear3).cpu().numpy(), parents_op))
                children_op = np.vstack((model(x_input.to(device), linear1=child_linear1, linear2=child_linear2, linear3=child_linear3).cpu().numpy(), children_op))

            # END IF 

        # END FOR

    # END FOR

    return parents_op, children_op


def calc_perf(prune_percent, parent_key, children_key):
    init_weights   = load_checkpoint('/z/home/madantrg/Pruning/results/STL10_ALEXNET_BATCH/0/logits_final.pkl')

    # Load Data
    trainloader, testloader = data_loader('STL10', 128)
 
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    model = alex(num_classes=10).to(device)
    model.load_state_dict(init_weights)
    model.eval()

    # Obtain Activations
    print("----------------------------------")
    print("Collecting activations from layers")

    act_start_time = time.time()
    p1_op, c1_op = activations(trainloader, model, device, parent_linear1=True, parent_linear2=False, parent_linear3=False, child_linear1=False, child_linear2=True, child_linear3=False)
    p2_op, c2_op = activations(trainloader, model, device, parent_linear1=False, parent_linear2=True, parent_linear3=False, child_linear1=False, child_linear2=False, child_linear3=True)
    act_end_time   = time.time()

    print("Time taken to collect activations is : %f seconds\n"%(act_end_time - act_start_time))

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (b)")

    # Algorithm 1(b)
    nlayers      = 2 
    children     = [8, 10]
    I_parent     = np.zeros((2, 10, p1_op.shape[1]))

    layer_start_time  = time.time()

    for num_layers in range(nlayers):
        for child in tqdm(range(children[num_layers])):
            for parent_1 in range(p1_op.shape[1]):
                g2 = np.setdiff1d(range(p1_op.shape[1]), parent_1)
                if num_layers == 0:
                    I_parent[num_layers, child, parent_1] = knn_mi(c1_op[:, child].reshape(-1,1), p1_op[:, parent_1].reshape(-1,1), p1_op[:, g2]) 

                else:
                    I_parent[num_layers, child, parent_1] = knn_mi(c2_op[:, child].reshape(-1,1), p2_op[:, parent_1].reshape(-1,1), p2_op[:, g2])


                ## END FOR
 
            # END FOR 

        # END FOR

    # END FOR

    layer_end_time = time.time()
    print('Time to calculate dependencies is %f seconds'%(layer_end_time - layer_start_time))
    print('Time to calculate dependencies for one child is %f seconds'%((layer_end_time - layer_start_time)/np.sum(children[:nlayers])))
    print('Time to calculate one I measure is %f seconds'%((layer_end_time - layer_start_time)/np.sum(children[:nlayers])/(p1_op.shape[1]*p1_op.shape[1] - p1_op.shape[1])))

    print("----------------------------------")
    print("Begin Pruning of Weights")
    #np.save('i_parent.npy', I_parent)
    #import pdb; pdb.set_trace()

    #I_parent = np.load('i_parent.npy')

    sorted_weights = np.sort(I_parent.reshape(-1)[np.where(I_parent.reshape(-1)!=0.)[0]])
    #sorted_weights = np.sort(I_parent[0].reshape(-1))
    #sorted_weights = np.sort(np.concatenate((I_parent[0][:8].reshape(-1), I_parent[1].reshape(-1))))
    cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
    cutoff_value   = sorted_weights[cutoff_index]
    print('Cutoff index %d wrt total number of elements %d' %(cutoff_index, sorted_weights.shape[0])) 
    print('Cutoff value %f' %(cutoff_value)) 


    for num_layers in range(len(parent_key)):
        parent_k   = parent_key[num_layers]
        children_k = children_key[num_layers]

        for child in range(children[num_layers]):
            for parent_1 in range(p1_op.shape[1]):
                if I_parent[num_layers, child, parent_1] < cutoff_value:
                    init_weights[children_k][child, parent_1] = 0.

                # END IF

            # END FOR

        # END FOR

    # END FOR

    model.load_state_dict(init_weights)
    acc = 100.*accuracy(model, testloader, device) 
    print('Accuracy of the pruned network on the 10000 test images: %f %%\n' %(acc))

    return acc

if __name__=='__main__':
    print "Calculation of performance change for one-shot pruning based on Mutual Information\n"

    perf = None
    parent_key   = ['linear1.weight', 'linear2.weight']
    children_key = ['linear2.weight', 'linear3.weight']

    for prune_percent in np.arange(0.0, 1.0, step=0.1):
        if perf is None:
            perf = [calc_perf(prune_percent, parent_key, children_key)]

        else:
            perf.append(calc_perf(prune_percent, parent_key, children_key))

        # END IF

    plt.plot(np.arange(0.0, 1.0, step=0.1), perf) 
    plt.xlabel('Ratio of weights pruned')
    plt.ylabel('Performance of AlexNet')
    plt.title('Comparison of performance variation when AlexNet is one-shot pruned, w/o retraining')
    plt.show()
