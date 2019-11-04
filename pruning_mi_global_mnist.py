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
import copy
import multiprocessing 
from sklearn.cluster import KMeans

#### Activation function ####
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


#### Sub-sample function ####
def sub_sample(activations, labels, num_samples_per_class=250):

    chosen_sample_idxs = []

    # Basic Implementation of Nearest Mean Classifier
    unique_labels = np.unique(labels)
    centroids     = np.zeros((len(unique_labels), activations.shape[1]))

    for idxs in range(len(unique_labels)):
        centroids[idxs] = np.mean(activations[np.where(labels==unique_labels[idxs])[0]], axis=0)
        chosen_idxs = np.argsort(np.linalg.norm(activations[np.where(labels==unique_labels[idxs])[0]] - centroids[idxs], axis=1))[:num_samples_per_class]
        chosen_sample_idxs.extend((np.where(labels==unique_labels[idxs])[0])[chosen_idxs].tolist())

    
    return activations[chosen_sample_idxs]

#### Conditional Mutual Information Computation For Alg. 1 (a) groups
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

#### Alg. 1 (a) groups
def alg1a_group(nlayers, children, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children):

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    pool = multiprocessing.Pool(10)

    for num_layers in tqdm(range(nlayers)):
        data = []

        for child in range(clusters_children[num_layers]):
            data.append([clusters[num_layers], c1_op, child, p1_op, num_layers, labels, labels_children])

        # END FOR 

        data = tuple(data)
        ret_values = pool.map(cmi, data)

        for child in range(clusters_children[num_layers]):
            I_parent[str(num_layers)][child,:] = ret_values[child]

        # END FOR 

    # END FOR

#### Alg. 1 (a) groups (Non parallelized version to test functionality and timing)
def alg1a_group_non(nlayers, children, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children):

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    for num_layers in range(nlayers):
        for child in range(clusters_children[num_layers]):
            for group_1 in range(clusters[num_layers]):
                import pdb; pdb.set_trace()
                for group_2 in tqdm(range(clusters[num_layers])):
                    if group_1 == group_2:
                        continue

                    I_parent[str(num_layers)][child, group_1] += mi(c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 

                # END FOR 

            # END FOR 

        # END FOR 

    # END FOR

#### Pruning and Saving Weights Function
def pruner(I_parent, prune_percent, parent_key, children_key, children, clusters, clusters_children, labels, labels_children, final_weights, model, testloader, device):

    # Create a copy
    init_weights   = copy.deepcopy(final_weights)

    sorted_weights = None
    mask_weights   = {}

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

        for child in range(clusters_children[num_layers]):
            for group_1 in range(clusters[num_layers]):
                if I_parent[str(num_layers)][child, group_1] <= cutoff_value:
                    for group_p in np.where(labels[str(num_layers)]==group_1)[0]:
                        for group_c in np.where(labels_children[str(num_layers)]==child)[0]:
                            init_weights[children_k][group_c, group_p] = 0.

                # END IF

            # END FOR

        # END FOR
        mask_weights[children_k] = np.ones(init_weights[children_k].shape)
        mask_weights[children_k][np.where(init_weights[children_k]==0)] = 0

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
   
    ## Save Mask
    np.save('logits_29_'+str(prune_percent*10)+'.npy', mask_weights)

 
    return acc, true_prune_percent


#### Main Code Executor 
def calc_perf(parent_key, children_key, clusters, clusters_children):



    #### Load Model ####
    init_weights   = load_checkpoint('/z/home/madantrg/Pruning/results/0/logits_29.pkl')

    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
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

    # Obtain Activations
    print("----------------------------------")
    print("Collecting activations from layers")

    act_start_time = time.time()
    p1_op = {}
    c1_op = {}

    unique_keys = np.unique(np.union1d(parent_key, children_key)).tolist()
    act         = {}
    lab         = {}

    for item_key in unique_keys:
        act[item_key], lab[item_key] = activations_mlp(trainloader, model, device, item_key)


    for item_idx in range(len(parent_key)):
        # Sub-sample activations
        p1_op[str(item_idx)] = copy.deepcopy(act[parent_key[item_idx]]) 
        c1_op[str(item_idx)] = copy.deepcopy(act[children_key[item_idx]])


    act_end_time   = time.time()

    print("Time taken to collect activations is : %f seconds\n"%(act_end_time - act_start_time))


    for idx in range(nlayers):
        labels[str(idx)]          = np.zeros((init_weights[children_key[idx]].shape[1],))
        labels_children[str(idx)] = np.zeros((init_weights[children_key[idx]].shape[0],))
        children[str(idx)]        = init_weights[children_key[idx]].shape[0]
        I_parent[str(idx)]        = np.zeros((clusters_children[idx], clusters[idx]))

        #### Compute Clusters/Groups ####
        print("----------------------------------")
        print("Begin Clustering Layers: %s and %s\n"%(parent_key[idx], children_key[idx]))

        # Parents
        if p1_op[str(idx)].shape[1] == clusters[idx]:
            labels[str(idx)] = np.arange(clusters[idx])

        else:
            #labels[str(idx)] = fclusterdata(X=whiten(p1_op[str(idx)].T), t=clusters[idx], criterion='maxclust', method='ward') - 1
            #kmeans = KMeans(init='k-means++', n_clusters=clusters[idx], n_init=100, max_iter=1000)
            #labels[str(idx)] = kmeans.fit_predict(p1_op[str(idx)].T)
            labels[str(idx)] = np.repeat(np.arange(clusters[idx]), labels[str(idx)].shape[0]/clusters[idx])

        # END IF

        # Children
        if c1_op[str(idx)].shape[1] == clusters_children[idx]:
            labels_children[str(idx)] = np.arange(clusters_children[idx])

        else:
            #labels_children[str(idx)] = fclusterdata(X=whiten(c1_op[str(idx)].T), t=clusters_children[idx], criterion='maxclust', method='ward') - 1
            #kmeans = KMeans(init='k-means++', n_clusters=clusters_children[idx], n_init=100, max_iter=1000)
            #labels_children[str(idx)] = kmeans.fit_predict(c1_op[str(idx)].T)
            labels_children[str(idx)] = np.repeat(np.arange(clusters_children[idx]), labels_children[str(idx)].shape[0]/clusters_children[idx])

        # END IF

    # END FOR        

    for item_idx in range(len(parent_key)):
        # Sub-sample activations
        p1_op[str(item_idx)] = sub_sample(copy.deepcopy(act[parent_key[item_idx]]),   lab[parent_key[item_idx]])
        c1_op[str(item_idx)] = sub_sample(copy.deepcopy(act[children_key[item_idx]]), lab[parent_key[item_idx]])

    del act, lab

    alg1a_group(nlayers, children, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children)


    print("----------------------------------")
    print("Begin Pruning of Weights")

    perf         = None
    prune_per    = None

    for prune_percent in np.arange(0.0, 1.0, step=0.1):
        acc, true_prune_percent = pruner(I_parent, prune_percent, parent_key, children_key, children, clusters, clusters_children, labels, labels_children, init_weights, model, testloader, device)
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

    perf              = None
    prune_per         = None
    parent_key        = ['fc1.weight','fc2.weight']
    children_key      = ['fc2.weight','fc3.weight']
    alg               = '1a_group'
    clusters          = [10, 10]#,150]
    clusters_children = [10, 10]#,20]

    perf, prune_per = calc_perf(parent_key, children_key, clusters, clusters_children)
