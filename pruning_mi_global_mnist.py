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


# Custom Imports
from pruning_utils import activations_mlp, sub_sample
from mi_estimation import *


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
def alg1a_group(nlayers, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children):

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
def alg1a_group_non(nlayers, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children):

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    for num_layers in range(nlayers):
        for child in range(clusters_children[num_layers]):
            for group_1 in range(clusters[num_layers]):
                for group_2 in tqdm(range(clusters[num_layers])):
                    if group_1 == group_2:
                        continue

                    I_parent[str(num_layers)][child, group_1] += mi(c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 

                # END FOR 

            # END FOR 

        # END FOR 

    # END FOR


#### Main Code Executor 
def calc_perf(parent_key, children_key, clusters, clusters_children, load_weights, save_data_dir):


    #### Load Model ####
    init_weights   = load_checkpoint(load_weights)

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

    alg1a_group(nlayers, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children)

    np.save(save_data_dir+'/I_parent.npy', I_parent)
    np.save(save_data_dir+'/Labels.npy', labels)
    np.save(save_data_dir+'/Labels_children.npy', labels_children)


if __name__=='__main__':
    print "Calculation of performance change for one-shot pruning based on Mutual Information"

    parent_key        = ['fc1.weight','fc2.weight']
    children_key      = ['fc2.weight','fc3.weight']
    alg               = '1a_group'
    clusters          = [10, 10]
    clusters_children = [10, 10]

    load_weights  = '/z/home/madantrg/Pruning/results/0/logits_best.pkl'
    save_data_dir = '/z/home/madantrg/Pruning/results/0/'

    calc_perf(parent_key, children_key, clusters, clusters_children, load_weights, save_data_dir)

    print('Code Execution Complete')
