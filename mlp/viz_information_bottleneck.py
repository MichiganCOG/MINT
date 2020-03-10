import time
import copy
import keras
import torch
import random
import argparse
import multiprocessing

import numpy             as np
import torch.nn          as nn
import matplotlib.pyplot as plt
import keras.backend     as K

from tqdm                    import tqdm

# Custom Imports
from data_handler            import data_loader
from utils                   import activations, sub_sample_uniform, mi
from utils                   import save_checkpoint, load_checkpoint, accuracy, mi
from scipy.special           import logsumexp
from model                   import MLP           as mlp 
from scipy.spatial.distance  import pdist
from scipy.spatial.distance  import squareform

# Fixed Backend To Force Dataloader To Be Consistent
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

nats2bits      = 1.0/np.log(2) 
noise_variance = 1e-1

def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    #x2 = np.expand_dims(np.sum(X ** 2, 1), 1)
    #dists = x2 + x2.T - 2*np.matmul(X, X.T)

    #alt_dists = pdist(X) 
    #dists     = squareform(alt_dists)

    #import pdb; pdb.set_trace()
    X = K.constant(X)
    x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    dists = x2 + K.transpose(x2) - 2*K.dot(X, K.transpose(X))
    return dists

def get_shape(x):
    dims = K.cast( K.shape(x)[1], K.floatx() ) 
    N    = K.cast( K.shape(x)[0], K.floatx() )
    #dims = float(x.shape[1]) 
    #N    = float(x.shape[0])
    return dims, N

def entropy_estimator_kl(x, var=1e-1):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    #normconst = (dims/2.)*np.log(2*np.pi*var)
    #lprobs = logsumexp(-dists2, axis=1) - np.log(N) - normconst
    #h = -np.mean(lprobs)
    
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
    h = -K.mean(lprobs)
    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)










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
def calc_perf(model, dataset, parent_key, weights_dir, cores, dims):


    #### Load Model ####
    init_weights   = load_checkpoint(weights_dir)#+'logits_best.pkl')

    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    if model == 'mlp':
        model = mlp(num_classes=dims).to(device)

    else:
        print('Invalid model selected')

    # END IF

    model.load_state_dict(init_weights)
    model.eval()

    #### Load Data ####
    trainloader, testloader, extraloader = data_loader(dataset, 64)
 
    nlayers         = len(parent_key)

    # Obtain Activations
    print("----------------------------------")
    print("Collecting activations from layers")

    act_start_time = time.time()
    p1_op = {}

    unique_keys = parent_key

    act         = {}
    lab         = {}

    for item_key in unique_keys:
        act[item_key], lab[item_key] = activations(testloader, model, device, item_key)

    for item_idx in range(len(parent_key)):
        # Sub-sample activations
        p1_op[str(item_idx)] = copy.deepcopy(act[parent_key[item_idx]]) 


    act_end_time   = time.time()

    print("Time taken to collect activations is : %f seconds\n"%(act_end_time - act_start_time))

    collector = lab[parent_key[0]]
    #for data, target in testloader:
    #    collector.extend(target.numpy().tolist())

    collect_idxs = {}
    for dim_idx in range(dims):
        collect_idxs[str(dim_idx)] = np.where(np.array(collector) == dim_idx)[0].tolist()


    h_upper    = np.zeros(nlayers)
    hM_given_X = np.zeros(nlayers)
    hM_given_Y = np.zeros(nlayers)

    for idx in range(nlayers):
        activity = p1_op[str(idx)]
        h_upper[idx] = entropy_estimator_kl(activity, noise_variance)

        hM_given_X[idx] = kde_condentropy(activity, noise_variance)
    
        #hM_given_Y_upper = 0.
        for dim_idx in range(dims):
                hcond_upper = entropy_estimator_kl(activity[collect_idxs[str(dim_idx)],:], noise_variance)
                hM_given_Y[idx] += 1./dims * hcond_upper
    


    #for item_idx in range(len(parent_key)):
    #    # Sub-sample activations
    #    p1_op[str(item_idx)] = sub_sample_uniform(copy.deepcopy(act[parent_key[item_idx]]),   lab[parent_key[item_idx]], num_samples_per_class=samples_per_class)
    #    c1_op[str(item_idx)] = sub_sample_uniform(copy.deepcopy(act[children_key[item_idx]]), lab[parent_key[item_idx]], num_samples_per_class=samples_per_class)

    del act, lab

    #alg1b_group(nlayers, I_parent, p1_op, c1_op, labels, labels_children, clusters, clusters_children, cores)

    #np.save('results/'+weights_dir+'I_parent_'+name_postfix+'.npy', I_parent)
    #np.save('results/'+weights_dir+'Labels_'+name_postfix+'.npy', labels)
    #np.save('results/'+weights_dir+'Labels_children_'+name_postfix+'.npy', labels_children)

    return h_upper, hM_given_X, hM_given_Y 

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

    args = parser.parse_args()

    #print('Selected key id is %d'%(args.key_id))

    #parents  = ['input',      'fc1.weight', 'fc2.weight', 'fc3.weight']
    #parents  = ['fc1.weight', 'fc2.weight']
    parents  = ['fc1.weight', 'fc2.weight', 'fc3.weight', 'fc4.weight']

    #calc_perf(args.model, args.dataset, [parents[args.key_id-1]], [children[args.key_id-1]], args.weights_dir, args.cores, args.dims)
    h_upper, hM_given_X, hM_given_Y = calc_perf(args.model, args.dataset, parents, args.weights_dir, args.cores, args.dims)

    for nlayer in range(len(parents)):
        print('Layer %s upper: MI(X;M) = %f, MI(Y;M) = %f'%(parents[nlayer], (h_upper-hM_given_X)[nlayer]*nats2bits, (h_upper-hM_given_Y)[nlayer]*nats2bits ))

    print('Code Execution Complete')
