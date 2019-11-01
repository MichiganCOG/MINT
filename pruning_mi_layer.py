import torch
import copy
import multiprocessing

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

def activations(data_loader, model, device, item_key):
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

       
    clusters, c1_op, child, p1_op, num_layers, labels, labels_children = data 
    I_value = np.zeros((clusters,))

    for group_1 in range(clusters):
        for group_2 in range(clusters):
            if group_1 == group_2:
                continue
            #I_value[group_1] += knn_mi(c1_op[str(num_layers)][:, child].reshape(-1,1), p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 
            I_value[group_1] += knn_mi(c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 

        # END FOR
 
    # END FOR 

    return I_value 

def alg1a_group_non(nlayers, children, I_parent, p1_op, c1_op, labels, clusters=2):

    print("----------------------------------")
    print("Begin Clustering to decide groups")

    # Parents
    for loop_idx in [len(labels.keys())-1]:
        # K-means
        #temp, labels[str(loop_idx)] = kmeans2(data=whiten(p1_op[str(loop_idx)].T), k=clusters, minit='points', iter=100)
        # Hierarchical
        labels[str(loop_idx)] = fclusterdata(X=whiten(p1_op[str(loop_idx)].T), t=clusters[loop_idx], criterion='maxclust') - 1

    labels_children = {}
    # Children
    for loop_idx in [len(labels.keys())-1]:
        # K-means
        #temp, labels[str(loop_idx)] = kmeans2(data=whiten(p1_op[str(loop_idx)].T), k=clusters, minit='points', iter=100)
        # Hierarchical
        labels_children[str(loop_idx)] = fclusterdata(X=whiten(c1_op[str(loop_idx)].T), t=clusters[loop_idx], criterion='maxclust') - 1

    #layer_start_time  = time.time()

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    data = []
    for num_layers in [len(labels.keys())-1]:
        for child in tqdm(range(children[str(num_layers)])):
        #for child in tqdm(range(clusters[num_layers])):
            for group_1 in range(clusters[num_layers]):
                for group_2 in range(clusters[num_layers]):
                    if group_1 == group_2:
                        continue
                    #I_value[group_1] += knn_mi(c1_op[str(num_layers)][:, child].reshape(-1,1), p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 
                    import pdb; pdb.set_trace()
                    I_parent[str(num_layers)][child, group_1] += knn_mi(c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 

        # END FOR 

    # END FOR

def alg1a_group(nlayers, children, I_parent, p1_op, c1_op, labels, clusters=2):

    print("----------------------------------")
    print("Begin Clustering to decide groups")

    # Parents
    for loop_idx in [len(labels.keys())-1]:
        # K-means
        #temp, labels[str(loop_idx)] = kmeans2(data=whiten(p1_op[str(loop_idx)].T), k=clusters, minit='points', iter=100)
        # Hierarchical
        labels[str(loop_idx)] = fclusterdata(X=whiten(p1_op[str(loop_idx)].T), t=clusters[loop_idx], criterion='maxclust') - 1

    labels_children = {}
    # Children
    for loop_idx in [len(labels.keys())-1]:
        # K-means
        #temp, labels[str(loop_idx)] = kmeans2(data=whiten(p1_op[str(loop_idx)].T), k=clusters, minit='points', iter=100)
        # Hierarchical
        labels_children[str(loop_idx)] = fclusterdata(X=whiten(c1_op[str(loop_idx)].T), t=clusters[loop_idx], criterion='maxclust') - 1

    #layer_start_time  = time.time()

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    pool = multiprocessing.Pool(6)

    data = []
    for num_layers in tqdm([len(labels.keys())-1]):
        #for child in range(children[str(num_layers)]):
        for child in range(clusters[num_layers]):
            data.append([clusters[num_layers], c1_op, child, p1_op, num_layers, labels, labels_children])

        # END FOR 

        data = tuple(data)
        ret_values = pool.map(cmi, data)

        for child in range(children[str(num_layers)]):
            I_parent[str(num_layers)][child,:] = ret_values[child]

        # END FOR 


    # END FOR


def pruner(I_parent, prune_percent, parent_key, children_key, children, clusters, labels, final_weights, model, testloader, device, ret=False):

    # Create a copy
    init_weights   = copy.deepcopy(final_weights)

    sorted_weights = None

    # Flatten I_parent dictionary
    for looper_idx in [len(parent_key)-1]:
        if sorted_weights is None:
            sorted_weights = I_parent[str(looper_idx)].reshape(-1)

        else:
            sorted_weights =  np.concatenate((sorted_weights, I_parent[str(looper_idx)].reshape(-1)))

    sorted_weights = np.sort(sorted_weights)
    cutoff_index   = np.round(prune_percent * sorted_weights.shape[0]).astype('int')
    cutoff_value   = sorted_weights[cutoff_index]

    if ret:
        print('Cutoff index %d wrt total number of elements %d' %(cutoff_index, sorted_weights.shape[0])) 
        print('Cutoff value %f' %(cutoff_value)) 



    for num_layers in [len(parent_key)-1]:
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

    if ret:
        print('Accuracy of the pruned network on the 10000 test images: %f %%\n' %(acc))
   
    if ret:
        return acc, true_prune_percent, init_weights

    else:
        return acc, true_prune_percent

    # END IF

def calc_perf(parent_key, children_key, alg, clusters):
    init_weights   = load_checkpoint('/z/home/madantrg/Pruning/results/STL10_ALEXNET_BATCH/0/logits_final.pkl')

    # Load Data
    trainloader, testloader = data_loader('STL10', 128)
 
    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    model = alex(num_classes=10).to(device)
    model.load_state_dict(init_weights)
    model.eval()
    
    # Sub-sample dataset if > 5k samples
    ss_idxs = []
    for looper_idx in np.arange(0,60000,6000):
        ss_idxs.extend(np.random.choice(np.arange(looper_idx, looper_idx+6000), 500, replace=False).tolist())

    ss_idxs = np.array(ss_idxs).astype('int32')

    nlayers      = len(parent_key)
    labels       = {}
    children     = {}
    I_parent     = {}

    for num_layers in range(len(parent_key)):

        # Obtain Activations
        print("----------------------------------")
        print("Collecting activations from layers")

        act_start_time = time.time()
        p1_op = {}
        c1_op = {}

        unique_keys = np.unique(np.union1d(parent_key[num_layers], children_key[num_layers])).tolist()
        act         = {}

        for item_key in unique_keys:
            act[item_key] = activations(trainloader, model, device, item_key)

        p1_op[str(num_layers)] = act[parent_key[num_layers]].copy()
        c1_op[str(num_layers)] = act[children_key[num_layers]].copy()

        del act

        act_end_time   = time.time()

        print("Time taken to collect activations is : %f seconds\n"%(act_end_time - act_start_time))

        print("----------------------------------")

        labels[str(num_layers)]   = np.zeros((init_weights[children_key[num_layers]].shape[1],))
        children[str(num_layers)] = init_weights[children_key[num_layers]].shape[0]
        I_parent[str(num_layers)] = np.zeros((init_weights[children_key[num_layers]].shape[0], clusters[num_layers]))

        alg1a_group_non(1, children, I_parent, p1_op, c1_op, labels, clusters)


        print("----------------------------------")
        print("Begin Pruning of Weights")

        perf         = None
        prune_per    = None
        steps        = 0.05


        for prune_percent in np.arange(0.0, 1.0, step=steps):
            acc, true_prune_percent = pruner(I_parent, prune_percent, parent_key[:num_layers+1], children_key[:num_layers+1], children, clusters, labels, init_weights, model, testloader, device, False)
            if perf is None:
                perf      = [acc]
                prune_per = [true_prune_percent]

            else:
                perf.append(acc)
                prune_per.append(true_prune_percent)

            # END IF

        acc, true_prune_percent, init_weights = pruner(I_parent, np.arange(0.0, 1.0, step=steps)[np.where(np.array(perf) == perf[np.where(np.array(perf) >= perf[0])[0][-1]])[0][-1]], parent_key[:num_layers+1], children_key[:num_layers+1], children, clusters, labels, init_weights, model, testloader, device, True)


    return perf, prune_per

if __name__=='__main__':
    print "Calculation of performance change for one-shot pruning based on Mutual Information"

    perf         = None
    prune_per    = None
    parent_key   = ['conv3.weight']#[,'conv2.weight','conv3.weight','conv4.weight','conv5.weight',  'linear1.weight','linear2.weight']#['linear1.weight','linear2.weight']#
    children_key = ['conv4.weight']#[,'conv3.weight','conv4.weight','conv5.weight','linear1.weight','linear2.weight','linear3.weight']#['linear2.weight','linear3.weight']#
    alg          = '1a_group'
    clusters     = [4,4]
    #clusters     = [15,30,40,30,30,4,4]#[4,4]#

    perf, prune_per = calc_perf(parent_key, children_key, alg, clusters)
