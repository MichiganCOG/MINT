import time
import copy
import torch
import argparse
import multiprocessing

import numpy             as np
import torch.nn          as nn
import matplotlib.pyplot as plt

from data_handler            import data_loader
from utils                   import save_checkpoint, load_checkpoint, accuracy, mi
from tqdm                    import tqdm

from models                  import alexnet       as alex
from models                  import MLP           as mlp 
from models                  import VGG16_bn      as vgg 
from models                  import Resnet56_A    as resnet56_a
from models                  import Resnet56_B    as resnet56_b 


# Custom Imports
from utils import activations, sub_sample_uniform, mi

# Global Variable


#### Conditional Mutual Information Computation For Alg. 1 (a) groups
def cmi(data):
    clusters, c1_op, child, p1_op, num_layers, labels, labels_children = data 
    I_value = np.zeros((clusters,))

    for group_1 in range(clusters):
        I_value[group_1] += mi(c1_op[str(num_layers)][:, np.where(labels_children[str(num_layers)]==child)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]!=group_1)[0]]) 

    # END FOR 


    return I_value 

#### Alg. 1 (b) groups
def alg1b_group(nlayers, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children, cores):

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (b) Group")

    pool = multiprocessing.Pool(cores)

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


#### Main Code Executor 
def calc_perf(model, dataset, parent_key, children_key, clusters, clusters_children, weights_dir, cores, name_postfix, samples_per_class, dims):


    ##### Load Model ####
    #init_weights   = load_checkpoint(weights_dir+'logits_best.pkl')

    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    if model == 'mlp':
        model = mlp(num_classes=dims).to(device)

    elif model == 'alexnet':
        model = alex(num_classes=dims).to(device)
    
    elif model == 'vgg':
        model = vgg(num_classes=dims).to(device)
    
    elif model == 'resnet_a':
        model = resnet56_a(num_classes=dims).to(device)

    elif model == 'resnet_b':
        model = resnet56_b(num_classes=dims).to(device)

    else:
        print('Invalid model selected')

    init_weights   = load_checkpoint(weights_dir+'logits_best.pkl')
    model.load_state_dict(init_weights)
    model.eval()

    #### Load Data ####
    trainloader, testloader, extraloader = data_loader(dataset, 64)
 
    
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
        act[item_key], lab[item_key] = activations(extraloader, model, device, item_key)

    for item_idx in range(len(parent_key)):
        # Sub-sample activations
        p1_op[str(item_idx)] = copy.deepcopy(act[parent_key[item_idx]]) 
        c1_op[str(item_idx)] = copy.deepcopy(act[children_key[item_idx]])


    act_end_time   = time.time()

    print("Time taken to collect activations is : %f seconds\n"%(act_end_time - act_start_time))


    for idx in range(nlayers):
        if children_key[idx] == 'linear1.weight':
            labels[str(idx)]          = np.zeros((256,))
        else:
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
            labels[str(idx)] = np.repeat(np.arange(clusters[idx]), labels[str(idx)].shape[0]/clusters[idx])

        # END IF

        # Children
        if c1_op[str(idx)].shape[1] == clusters_children[idx]:
            labels_children[str(idx)] = np.arange(clusters_children[idx])

        else:
            labels_children[str(idx)] = np.repeat(np.arange(clusters_children[idx]), labels_children[str(idx)].shape[0]/clusters_children[idx])

        # END IF

    # END FOR        

    for item_idx in range(len(parent_key)):
        # Sub-sample activations
        p1_op[str(item_idx)] = sub_sample_uniform(copy.deepcopy(act[parent_key[item_idx]]),   lab[parent_key[item_idx]], num_samples_per_class=samples_per_class)
        c1_op[str(item_idx)] = sub_sample_uniform(copy.deepcopy(act[children_key[item_idx]]), lab[parent_key[item_idx]], num_samples_per_class=samples_per_class)

    del act, lab

    alg1b_group(nlayers, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children, cores)

    np.save(weights_dir+'/I_parent_'+name_postfix+'.npy', I_parent)
    np.save(weights_dir+'/Labels_'+name_postfix+'.npy', labels)
    np.save(weights_dir+'/Labels_children_'+name_postfix+'.npy', labels_children)


if __name__=='__main__':

    """"
    Sample Input values

    parent_key        = ['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight', 'conv10.weight','conv11.weight','conv12.weight','conv13.weight', 'linear1.weight']
    children_key      = ['conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight','conv10.weight','conv11.weight','conv12.weight','conv13.weight','linear1.weight', 'linear3.weight']
    alg               = '1a_group'
    clusters          = [8,8,8,8,8,8,8,8,8,8,8,8,8,8]
    clusters_children = [8,8,8,8,8,8,8,8,8,8,8,8,8,8]

    load_weights  = '/z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/logits_best.pkl'
    save_data_dir = '/z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/'
    """


    parser = argparse.ArgumentParser()

    parser.add_argument('--model',                type=str)
    parser.add_argument('--dataset',              type=str)
    parser.add_argument('--weights_dir',          type=str)
    parser.add_argument('--cores',                type=int)
    parser.add_argument('--dims',                 type=int, default=10)
    parser.add_argument('--key_id',               type=int)
    parser.add_argument('--samples_per_class',    type=int,      default=250)
    parser.add_argument('--children_key',         nargs='+',     type=str,       default=['conv2.weight'])
    parser.add_argument('--parent_clusters',      nargs='+',     type=int,       default=[8])
    parser.add_argument('--children_clusters',    nargs='+',     type=int,       default=[8])
    parser.add_argument('--name_postfix',         type=str)

    args = parser.parse_args()

    print('Selected key id is %d'%(args.key_id))

    args.key_id = args.key_id + 1

    if args.model == 'mlp':
        parents  = ['fc1.weight','fc2.weight']
        children = ['fc2.weight','fc3.weight']

    elif args.model == 'vgg':
        parents  = ['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight', 'conv10.weight','conv11.weight','conv12.weight','conv13.weight', 'linear1.weight']
        children = ['conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight','conv10.weight','conv11.weight','conv12.weight','conv13.weight','linear1.weight', 'linear3.weight']

    elif args.model == 'alexnet':
        parents  = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight',   'linear1.weight', 'linear2.weight']
        children = ['conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'linear1.weight', 'linear2.weight', 'linear3.weight']

 
    elif 'resnet' in args.model:
        parents  = ['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight', 'conv10.weight',
                    'conv11.weight','conv12.weight','conv13.weight','conv14.weight','conv15.weight','conv16.weight','conv17.weight','conv18.weight','conv19.weight', 'conv20.weight',
                    'conv21.weight','conv22.weight','conv23.weight','conv24.weight','conv25.weight','conv26.weight','conv27.weight','conv28.weight','conv29.weight', 'conv30.weight',
                    'conv31.weight','conv32.weight','conv33.weight','conv34.weight','conv35.weight','conv36.weight','conv37.weight','conv38.weight','conv39.weight', 'conv40.weight',
                    'conv41.weight','conv42.weight','conv43.weight','conv44.weight','conv45.weight','conv46.weight','conv47.weight','conv48.weight','conv49.weight', 'conv50.weight',
                    'conv51.weight','conv52.weight','conv53.weight','conv54.weight', 'conv55.weight']

        children = ['conv2.weight','conv3.weight','conv4.weight','conv5.weight','conv6.weight','conv7.weight','conv8.weight','conv9.weight', 'conv10.weight',
                    'conv11.weight','conv12.weight','conv13.weight','conv14.weight','conv15.weight','conv16.weight','conv17.weight','conv18.weight','conv19.weight', 'conv20.weight',
                    'conv21.weight','conv22.weight','conv23.weight','conv24.weight','conv25.weight','conv26.weight','conv27.weight','conv28.weight','conv29.weight', 'conv30.weight',
                    'conv31.weight','conv32.weight','conv33.weight','conv34.weight','conv35.weight','conv36.weight','conv37.weight','conv38.weight','conv39.weight', 'conv40.weight',
                    'conv41.weight','conv42.weight','conv43.weight','conv44.weight','conv45.weight','conv46.weight','conv47.weight','conv48.weight','conv49.weight', 'conv50.weight',
                    'conv51.weight','conv52.weight','conv53.weight','conv54.weight','conv55.weight', 'linear1.weight']

        if args.key_id >=20 and args.key_id <=37:
            if args.parent_clusters[0] == 8:
                args.parent_clusters[0] = 16
            else:
                args.parent_clusters[0] = 32

        if args.key_id >=38:
            if args.parent_clusters[0] == 8:
                args.parent_clusters[0] = 32
            else:
                args.parent_clusters[0] = 64

        if args.key_id >=19 and args.key_id <=36:
            if args.children_clusters[0] == 8:
                args.children_clusters[0] = 16
            else:
                args.children_clusters[0] = 32

        if args.key_id >=37:
            if args.children_clusters[0] == 8:
                args.children_clusters[0] = 32
            else:
                args.children_clusters[0] = 64

    if args.key_id ==len(parents):
        args.children_clusters = [10]
 
    calc_perf(args.model, args.dataset, [parents[args.key_id-1]], [children[args.key_id-1]], args.parent_clusters, args.children_clusters, args.weights_dir, args.cores, args.name_postfix +'_'+parents[args.key_id-1]+'_'+children[args.key_id-1], args.samples_per_class, args.dims)

    print('Code Execution Complete')