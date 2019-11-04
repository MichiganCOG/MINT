import torch
import copy
import multiprocessing

import numpy             as np
import matplotlib.pyplot as plt

from models                  import Alexnet            as alex
from models                  import MLP                as mlp 

from data_handler            import data_loader
from utils                   import save_checkpoint, load_checkpoint, accuracy
from hpmi                    import *
from tqdm                    import tqdm
from scipy.cluster.vq        import vq, kmeans2, whiten
from sklearn.decomposition   import PCA
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster         import MeanShift 

def activations_mlp(data_loader, model, device, item_key):
    parents_op  = None
    labels_op   = None

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data

            if parents_op is None:
                if 'fc1' in item_key:
                    parents_op  = model(x_input.to(device), fc1=True).cpu().numpy()
                elif 'fc2' in item_key:
                    parents_op  = model(x_input.to(device), fc2=True).cpu().numpy()
                elif 'fc3' in item_key:
                    parents_op  = model(x_input.to(device)).cpu().numpy()

                labels_op = y_label.numpy()

            else:
                if 'fc1' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), fc1=True).cpu().numpy(), parents_op))
                elif 'fc2' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), fc2=True).cpu().numpy(), parents_op))
                elif 'fc3' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device)).cpu().numpy(), parents_op))

                labels_op = np.hstack((labels_op, y_label.numpy()))

            # END IF 

        # END FOR

    # END FOR


    if len(parents_op.shape) > 2:
        parents_op  = np.mean(parents_op, axis=(2,3))

    return parents_op, labels_op



def sub_sample(activations, labels, num_samples_per_class=100):

    chosen_sample_idxs = []

    # Basic Implementation of Nearest Mean Classifier
    unique_labels = np.unique(labels)
    centroids     = np.zeros((len(unique_labels), activations.shape[1]))

    for idxs in range(len(unique_labels)):
        centroids[idxs] = np.mean(activations[np.where(labels==unique_labels[idxs])[0]], axis=0)
        chosen_idxs = np.argsort(np.linalg.norm(activations[np.where(labels==unique_labels[idxs])[0]] - centroids[idxs], axis=1))[:num_samples_per_class]
        chosen_sample_idxs.extend((np.where(labels==unique_labels[idxs])[0])[chosen_idxs].tolist())

    
    return activations[chosen_sample_idxs]

def cmi(data):
    clusters, c1_op, child, p1_op, num_layers, labels, labels_children = data 
    I_value = np.zeros((clusters,))

    for group_1 in range(clusters):
        for group_2 in range(clusters):
            if group_1 == group_2:
                continue

            I_value[group_1] += mi(c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 

    # END FOR 


    return I_value 

def cmi_group(data):
    clusters, c1_op, child, p1_op, num_layers, labels, labels_children, group_1 = data 
    I_value = 0. 

    for group_2 in range(clusters):
        if group_1 == group_2:
            continue

        I_value += (mi(c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]))/float(clusters-1)

    # END FOR 


    return I_value 

def alg1a_group_g_parallel(nlayers, children, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children):

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    pool = multiprocessing.Pool(6)

    for num_layers in [len(labels.keys())-1]:
        for child in tqdm(range(clusters_children[num_layers])):
            data = []
            for group_1 in range(clusters[num_layers]):
                data.append([clusters[num_layers], c1_op, child, p1_op, num_layers, labels, labels_children, group_1])

            # END FOR 

            data = tuple(data)
            ret_values = pool.map(cmi_group, data)

            for group_1 in range(clusters[num_layers]):
                I_parent[str(num_layers)][child,group_1] = ret_values[group_1]

            # END FOR
 
        # END FOR

    # END FOR

def alg1a_group_non(nlayers, children, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children):

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    pool = multiprocessing.Pool(6)
    data = []

    for num_layers in [len(labels.keys())-1]:
        for child in tqdm(range(clusters_children[num_layers])):
            for group_1 in range(clusters[num_layers]):
                for group_2 in range(clusters[num_layers]):
                    if group_1 == group_2:
                        continue

                    I_parent[str(num_layers)][child, group_1] += (mi(c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]])/float(clusters[num_layers]-1)) 
                    if I_parent[str(num_layers)][child, group_1] >= 0.75:
                        break

            # END FOR 

        # END FOR 

    # END FOR
    import pdb; pdb.set_trace()

def alg1a_group(nlayers, children, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children):

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    pool = multiprocessing.Pool(6)
    data = []

    for num_layers in tqdm([len(labels.keys())-1]):
        for child in range(clusters_children[num_layers]):
            data.append([clusters[num_layers], c1_op, child, p1_op, num_layers, labels, labels_children])

        # END FOR 

        data = tuple(data)
        ret_values = pool.map(cmi, data)

        for child in range(clusters_children[num_layers]):
            I_parent[str(num_layers)][child,:] = ret_values[child]

        # END FOR
 
    # END FOR

def pruner(I_parent, prune_percent, parent_key, children_key, children, clusters, clusters_children, labels, labels_children, final_weights, model, testloader, device, ret=False):

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

        for child in range(clusters_children[num_layers]):
            for group_1 in range(clusters[num_layers]):
                if I_parent[str(num_layers)][child, group_1] < cutoff_value:
                    for group_p in np.where(labels[str(num_layers)]==group_1)[0]:
                        for group_c in np.where(labels_children[str(num_layers)]==child)[0]:
                            init_weights[children_k][group_c, group_p] = 0.

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
            print('Percent weights remaining from %d layers is %d / %d'%(len(parent_key), valid_count, total_count))

    else:
        valid_count = len(np.where(init_weights[children_key[0]].reshape(-1)!= 0.0)[0])
        total_count = float(init_weights[children_key[0]].reshape(-1).shape[0])
   
        if ret:
            print('Percent weights remaining from %d layers is %f'%(len(parent_key), valid_count/total_count*100.))
            print('Percent weights remaining from %d layers is %d / %d'%(len(parent_key), valid_count, total_count))



    true_prune_percent = valid_count / float(total_count) * 100.

    model.load_state_dict(init_weights)
    acc = 100.*accuracy(model, testloader, device)

    if ret:
        print('Accuracy of the pruned network on test set is: %f %%\n' %(acc))
   
    if ret:
        return acc, true_prune_percent, init_weights

    else:
        return acc, true_prune_percent

    # END IF

def calc_perf(parent_key, children_key, alg, clusters, clusters_children):



    #### Load Model ####
    init_weights   = load_checkpoint('/z/home/madantrg/Pruning/results/0/logits_29.pkl')

    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = mlp(num_classes=10).to(device)
    model.load_state_dict(init_weights)
    model.eval()

    #### Load Data ####
    trainloader, testloader = data_loader('MNIST', 128)
 
    
    # Original Accuracy 
    acc = 100.*accuracy(model, testloader, device)
    print('Accuracy of the original network: %f %%\n' %(acc))


    nlayers         = len(parent_key)
    labels          = {}
    labels_children = {}
    children        = {}
    I_parent        = {}

    for num_layers in range(len(parent_key)):

        #### Collect Activations ####
        print("----------------------------------")
        print("Collecting activations from layers")

        act_start_time = time.time()
        p1_op = {}
        c1_op = {}

        unique_keys = np.unique(np.union1d(parent_key[num_layers], children_key[num_layers])).tolist()
        act         = {}
        lab         = {}

        for item_key in unique_keys:
            act[item_key], lab[item_key] = activations_mlp(trainloader, model, device, item_key)

        # Sub-sample activations
        p1_op[str(num_layers)] = sub_sample(copy.deepcopy(act[parent_key[num_layers]]), lab[parent_key[num_layers]])
        c1_op[str(num_layers)] = sub_sample(copy.deepcopy(act[children_key[num_layers]]), lab[parent_key[num_layers]])

        del act, lab

        act_end_time   = time.time()

        print("Time taken to collect activations is : %f seconds\n"%(act_end_time - act_start_time))

        labels[str(num_layers)]          = np.zeros((init_weights[children_key[num_layers]].shape[1],))
        labels_children[str(num_layers)] = np.zeros((clusters_children[num_layers],))
        children[str(num_layers)]        = init_weights[children_key[num_layers]].shape[0]
        I_parent[str(num_layers)]        = np.zeros((clusters_children[num_layers], clusters[num_layers]))

        #### Compute Clusters/Groups ####
        print("----------------------------------")
        print("Begin Clustering\n")

        # Parents
        if p1_op[str(num_layers)].shape[1] == clusters[num_layers]:
            labels[str(num_layers)] = np.arange(clusters[num_layers])

        else:
            labels[str(num_layers)] = fclusterdata(X=whiten(p1_op[str(num_layers)].T), t=clusters[num_layers], criterion='maxclust', method='weighted') - 1

        # Children
        if c1_op[str(num_layers)].shape[1] == clusters_children[num_layers]:
            labels_children[str(num_layers)] = np.arange(clusters_children[num_layers])

        else:
            labels_children[str(num_layers)] = fclusterdata(X=whiten(c1_op[str(num_layers)].T), t=clusters_children[num_layers], criterion='maxclust', method='weighted') - 1

        #### Begin MINT ####
        alg1a_group_g_parallel(1, children, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children)

        #### Begin Pruning and Performance Block ####
        print("----------------------------------")
        print("Begin Pruning of Weights")

        perf         = None
        prune_per    = None
        steps        = 0.05


        for prune_percent in np.arange(0.0, 1.0, step=steps):
            acc, true_prune_percent = pruner(I_parent, prune_percent, parent_key[:num_layers+1], children_key[:num_layers+1], children, clusters, clusters_children, labels, labels_children, init_weights, model, testloader, device, False)
            if perf is None:
                perf      = [acc]
                prune_per = [true_prune_percent]

            else:
                perf.append(acc)
                prune_per.append(true_prune_percent)

            # END IF

        acc, true_prune_percent, init_weights = pruner(I_parent, np.arange(0.0, 1.0, step=steps)[np.where(np.array(perf) == perf[np.where(np.array(perf) >= perf[0])[0][-1]])[0][-1]], parent_key[:num_layers+1], children_key[:num_layers+1], children, clusters, clusters_children, labels, labels_children, init_weights, model, testloader, device, True)


    return perf, prune_per

if __name__=='__main__':
    print "Calculation of performance change for one-shot pruning based on Mutual Information"

    perf              = None
    prune_per         = None
    parent_key        = ['fc1.weight','fc2.weight']
    children_key      = ['fc2.weight','fc3.weight']
    alg               = '1a_group'
    clusters          = [20, 20]#,150]
    clusters_children = [10, 10]#,20]

    perf, prune_per = calc_perf(parent_key, children_key, alg, clusters, clusters_children)
