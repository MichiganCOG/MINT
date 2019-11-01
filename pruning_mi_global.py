import torch
import copy
import multiprocessing 

import numpy             as np
import matplotlib.pyplot as plt

from data_handler          import data_loader
from models                import Alexnet        as alex
from utils                 import save_checkpoint, load_checkpoint, accuracy
from knn                   import *
from tqdm                  import tqdm
from scipy.cluster.vq      import vq, kmeans2, whiten
from sklearn.decomposition import PCA

def activations(data_loader, model, device, item_key):#, sub_sample_idxs):
    parents_op  = None

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data

            if parents_op is None:
                if 'conv1' in item_key:
                    parents_op  = model(x_input.to(device), conv1=True).cpu().numpy()
                elif 'conv2' in item_key:
                    parents_op  = model(x_input.to(device), conv2=True).cpu().numpy()
                elif 'conv3' in item_key:
                    parents_op  = model(x_input.to(device), conv3=True).cpu().numpy()
                elif 'conv4' in item_key:
                    parents_op  = model(x_input.to(device), conv4=True).cpu().numpy()
                elif 'conv5' in item_key:
                    parents_op  = model(x_input.to(device), conv5=True).cpu().numpy()
                elif 'linear1' in item_key:
                    parents_op  = model(x_input.to(device), linear1=True).cpu().numpy()
                elif 'linear2' in item_key:
                    parents_op  = model(x_input.to(device), linear2=True).cpu().numpy()
                elif 'linear3' in item_key:
                    parents_op  = model(x_input.to(device), linear3=True).cpu().numpy()

            else:
                if 'conv1' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), conv1=True).cpu().numpy(), parents_op))
                elif 'conv2' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), conv2=True).cpu().numpy(), parents_op))
                elif 'conv3' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), conv3=True).cpu().numpy(), parents_op))
                elif 'conv4' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), conv4=True).cpu().numpy(), parents_op))
                elif 'conv5' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), conv5=True).cpu().numpy(), parents_op))
                elif 'linear1' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), linear1=True).cpu().numpy(), parents_op))
                elif 'linear2' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), linear2=True).cpu().numpy(), parents_op))
                elif 'linear3' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), linear3=True).cpu().numpy(), parents_op))

            # END IF 

        # END FOR

    # END FOR


    if len(parents_op.shape) > 2:
        parents_op  = np.mean(parents_op, axis=(2,3))

    return parents_op

def cmi(data):

       
    clusters, c1_op, child, p1_op, num_layers, labels = data 
    I_value = np.zeros((clusters,))

    for group_1 in range(clusters):
        for group_2 in range(clusters):
            if group_1 == group_2:
                continue
            I_value[group_1] += knn_mi(c1_op[str(num_layers)][:, child].reshape(-1,1), p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 

        # END FOR
 
    # END FOR 

    return I_value 

def alg1a_group(nlayers, children, I_parent, p1_op, c1_op, labels, clusters):

    print("----------------------------------")
    print("Begin Clustering to decide groups")

    for loop_idx in range(len(labels.keys())):
        temp, labels[str(loop_idx)] = kmeans2(data=whiten(p1_op[str(loop_idx)].T), k=clusters[loop_idx], minit='points')


    #layer_start_time  = time.time()

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    pool = multiprocessing.Pool(6)

    for num_layers in tqdm(range(nlayers)):
        data = []

        for child in range(children[str(num_layers)]):
            data.append([clusters[num_layers], c1_op, child, p1_op, num_layers, labels])

        # END FOR 

        data = tuple(data)
        ret_values = pool.map(cmi, data)

        for child in range(children[str(num_layers)]):
            I_parent[str(num_layers)][child,:] = ret_values[child]

        # END FOR 

    # END FOR

def pruner(I_parent, prune_percent, parent_key, children_key, children, clusters, labels, final_weights, model, testloader, device):

    # Create a copy
    init_weights   = copy.deepcopy(final_weights)

    sorted_weights = None
    # Flatten I_parent dictionary
    for looper_idx in range(len(I_parent.keys())):
        if sorted_weights is None:
            sorted_weights = I_parent[str(looper_idx)].reshape(-1)
        else:
            sorted_weights =  np.concatenate((sorted_weights, I_parent[str(looper_idx)].reshape(-1)))

    sorted_weights = np.sort(sorted_weights)
    cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
    cutoff_value   = sorted_weights[cutoff_index]
    print('Cutoff index %d wrt total number of elements %d' %(cutoff_index, sorted_weights.shape[0])) 
    print('Cutoff value %f' %(cutoff_value)) 



    for num_layers in range(len(parent_key)):
        parent_k   = parent_key[num_layers]
        children_k = children_key[num_layers]

        for child in range(children[str(num_layers)]):
            for group_1 in range(clusters[num_layers]):
                if I_parent[str(num_layers)][child, group_1] < cutoff_value:
                    for group in np.where(labels[str(num_layers)]==group_1)[0]:
                        init_weights[children_k][child, group] = 0.

                # END IF

            # END FOR

        # END FOR

    # END FOR

    if len(parent_key) > 1:
        total_count = 0
        valid_count = 0
        for num_layers in range(len(parent_key)):
            total_count += init_weights[children_key[num_layers]].reshape(-1).shape[0]
            valid_count += len(np.where(init_weights[children_key[num_layers]].reshape(-1)!=0.)[0])
        
        print('Percent weights remaining from %d layers is %f'%(len(parent_key), valid_count/float(total_count)*100.))

    else:
        valid_count = len(np.where(init_weights[children_key[0]].reshape(-1)!= 0.0)[0])
        total_count = float(init_weights[children_key[0]].reshape(-1).shape[0])
        print('Percent weights remaining from %d layers is %f'%(len(parent_key), valid_count/total_count*100.))



    true_prune_percent = valid_count / float(total_count) * 100.

    model.load_state_dict(init_weights)
    acc = 100.*accuracy(model, testloader, device) 
    print('Accuracy of the pruned network on the 10000 test images: %f %%\n' %(acc))
    
    return acc, true_prune_percent



def calc_perf(parent_key, children_key, clusters):
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
    p1_op = {}
    c1_op = {}

    unique_keys = np.unique(np.union1d(parent_key, children_key)).tolist()
    act         = {}

    for item_key in unique_keys:
        act[item_key] = activations(trainloader, model, device, item_key)


    for item_idx in range(len(parent_key)):
        p1_op[str(item_idx)] = act[parent_key[item_idx]].copy()
        c1_op[str(item_idx)] = act[children_key[item_idx]].copy()

    del act

    act_end_time   = time.time()

    print("Time taken to collect activations is : %f seconds\n"%(act_end_time - act_start_time))

    print("----------------------------------")

    nlayers      = len(parent_key)
    labels       = {}
    children     = {}
    I_parent     = {}

    for idx in range(nlayers):
        labels[str(idx)]   = np.zeros((init_weights[children_key[idx]].shape[1],))
        children[str(idx)] = init_weights[children_key[idx]].shape[0]
        I_parent[str(idx)] = np.zeros((init_weights[children_key[idx]].shape[0], clusters[idx]))

    # END FOR        


    alg1a_group(nlayers, children, I_parent, p1_op, c1_op, labels, clusters)


    print("----------------------------------")
    print("Begin Pruning of Weights")

    perf         = None
    prune_per    = None

    for prune_percent in np.arange(0.0, 1.0, step=0.05):
        acc, true_prune_percent = pruner(I_parent, prune_percent, parent_key, children_key, children, clusters, labels, init_weights, model, testloader, device)
        if perf is None:
            perf      = [acc]
            prune_per = [true_prune_percent]

        else:
            perf.append(acc)
            prune_per.append(true_prune_percent)

        # END IF

    print(prune_per)
    print(perf)


    return perf, prune_per

if __name__=='__main__':
    print "Calculation of performance change for one-shot pruning based on Mutual Information"

    perf         = None
    prune_per    = None
    parent_key   = ['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight',  'linear1.weight','linear2.weight']#['linear1.weight','linear2.weight']#
    children_key = ['conv2.weight','conv3.weight','conv4.weight','conv5.weight','linear1.weight','linear2.weight','linear3.weight']#['linear2.weight','linear3.weight']#
    clusters     = [15,30,40,30,30,4,4]#[4,4]# 

    perf, prune_per = calc_perf(parent_key, children_key, clusters)

    plt.plot(prune_per, perf) 
    plt.xlabel('Ratio of weights pruned')
    plt.ylabel('Performance of AlexNet')
    plt.title('Comparison of performance variation when AlexNet is one-shot pruned, w/o retraining')
    plt.show()
