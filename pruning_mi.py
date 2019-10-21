import torch
import copy

import numpy             as np
import matplotlib.pyplot as plt

from data_handler          import data_loader
from models                import Alexnet        as alex
from utils                 import save_checkpoint, load_checkpoint, accuracy
from knn                   import *
from tqdm                  import tqdm
from scipy.cluster.vq      import vq, kmeans2, whiten
from sklearn.decomposition import PCA


def activations(data_loader, model, device, parent_conv1, parent_conv2, parent_conv3, parent_conv4, parent_conv5, parent_linear1, parent_linear2, parent_linear3, child_conv1, child_conv2, child_conv3, child_conv4, child_conv5, child_linear1, child_linear2, child_linear3):
    parents_op  = None
    children_op = None

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data

            if parents_op is None:
                parents_op  = model(x_input.to(device), conv1=parent_conv1, conv2=parent_conv2, conv3=parent_conv3, conv4=parent_conv4, conv5=parent_conv5, linear1=parent_linear1, linear2=parent_linear2, linear3=parent_linear3).cpu().numpy()
                children_op = model(x_input.to(device), conv1=child_conv1, conv2=child_conv2, conv3=child_conv3, conv4=child_conv4, conv5=child_conv5, linear1=child_linear1, linear2=child_linear2, linear3=child_linear3).cpu().numpy()

            else:
                parents_op  = np.vstack((model(x_input.to(device), conv1=parent_conv1, conv2=parent_conv2, conv3=parent_conv3, conv4=parent_conv4, conv5=parent_conv5, linear1=parent_linear1, linear2=parent_linear2, linear3=parent_linear3).cpu().numpy(), parents_op))
                children_op = np.vstack((model(x_input.to(device), conv1=child_conv1, conv2=child_conv2, conv3=child_conv3, conv4=child_conv4, conv5=child_conv5, linear1=child_linear1, linear2=child_linear2, linear3=child_linear3).cpu().numpy(), children_op))

            # END IF 

        # END FOR

    # END FOR


    if len(parents_op.shape) > 2:
        parents_op  = np.mean(parents_op, axis=(2,3))

    if len(children_op.shape) > 2:
        children_op = np.mean(children_op, axis=(2,3))


        #samples, batch, h, w = parents_op.shape
        #parents_op  = np.vstack(parents_op).reshape(samples*batch, -1)
        #pca_parents = PCA(n_components=100)
        #parents_op  = pca_parents.fit_transform(parents_op).reshape(samples, batch, -1)

        #samples, batch, h, w = children_op.shape
        #children_op  = np.vstack(children_op).reshape(samples*batch, -1)
        #pca_children = PCA(n_components=100)
        #children_op  = pca_children.fit_transform(children_op).reshape(samples, batch, -1)

        #parents_op   = parents_op.reshape((samples, batch, -1))
        #children_op  = children_op.reshape((samples, batch, -1))

    return parents_op, children_op


def alg1a(nlayers, children, I_parent, p1_op, p2_op, c1_op, c2_op):
    print("Begin Execution of Algorithm 1 (a)")

    layer_start_time  = time.time()

    for num_layers in range(nlayers):
        for child in tqdm(range(children[num_layers])):
            for parent_1 in range(p1_op.shape[1]):
                for parent_2 in range(p1_op.shape[1]):
                    if parent_1 == parent_2:
                        continue

                    if num_layers == 0:
                        I_value = knn_mi(c1_op[:, child].reshape(-1,1), p1_op[:, parent_1].reshape(-1,1), p1_op[:, parent_2].reshape(-1,1)) 
                        I_parent[num_layers, child, parent_1] += I_value

                    else:
                        I_value = knn_mi(c2_op[:, child].reshape(-1,1), p2_op[:, parent_1].reshape(-1,1), p2_op[:, parent_2].reshape(-1,1))
                        I_parent[num_layers, child, parent_1] += I_value

                # END FOR
 
            # END FOR 

        # END FOR

    # END FOR

    layer_end_time = time.time()
    print('Time to calculate dependencies is %f seconds'%(layer_end_time - layer_start_time))
    print('Time to calculate dependencies for one child is %f seconds'%((layer_end_time - layer_start_time)/np.sum(children[:nlayers])))
    print('Time to calculate one I measure is %f seconds'%((layer_end_time - layer_start_time)/np.sum(children[:nlayers])/(p1_op.shape[1]*p1_op.shape[1] - p1_op.shape[1])))

def alg1b(nlayers, children, I_parent, p1_op, p2_op, c1_op, c2_op):
    print("Begin Execution of Algorithm 1 (b)")

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

def alg1a_group(nlayers, children, I_parent, p1_op, c1_op, labels, clusters=2):

    print("----------------------------------")
    print("Begin Clustering to decide groups")

    for loop_idx in range(len(labels.keys())):
        temp, labels[str(loop_idx)] = kmeans2(data=whiten(p1_op[str(loop_idx)].T), k=clusters, minit='points')


    #centroids, labels[0] = kmeans2(data=whiten(p1_op.T), k=clusters, minit='points')
    ##centroids, labels[0] = kmeans2(data=whiten(p1_op.transpose((1,0,2)).reshape(p1_op.shape[1],-1)), k=clusters, minit='points')
    #if labels.shape[0] > 1:
    #    centroids, labels[1] = kmeans2(data=whiten(p2_op.T), k=clusters, minit='points')

    #layer_start_time  = time.time()

    print("----------------------------------")
    print("Begin Execution of Algorithm 1 (a) Group")

    for num_layers in range(nlayers):
        for child in tqdm(range(children[str(num_layers)])):
            for group_1 in range(clusters):
                for group_2 in range(clusters):
                    if group_1 == group_2:
                        continue

                    #if num_layers == 0:
                    I_value = knn_mi(c1_op[str(num_layers)][:, child].reshape(-1,1), p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_1)[0]], p1_op[str(num_layers)][:, np.where(labels[str(num_layers)]==group_2)[0]]) 
                        #I_value = knn_mi(c1_op[:, child],  np.hstack(p1_op[:, np.where(labels[num_layers]==group_1)[0]].transpose((1,0,2))),  np.hstack(p1_op[:, np.where(labels[num_layers]==group_2)[0]].transpose((1,0,2)))) 
                    I_parent[str(num_layers)][child, group_1] += I_value

                    #else:
                    #    I_value = knn_mi(c2_op[:, child].reshape(-1,1), p2_op[:, np.where(labels[num_layers]==group_1)[0]], p2_op[:, np.where(labels[num_layers]==group_2)[0]])
                    #    I_parent[num_layers, child, group_1] += I_value

                # END FOR
 
            # END FOR 

        # END FOR

    # END FOR

    #layer_end_time = time.time()
    #print('Time to calculate dependencies is %f seconds'%(layer_end_time - layer_start_time))
    #print('Time to calculate dependencies for one child is %f seconds'%((layer_end_time - layer_start_time)/np.sum(children[:nlayers])))
    #print('Time to calculate one I measure is %f seconds'%((layer_end_time - layer_start_time)/np.sum(children[:nlayers])/(p1_op.shape[1]*p1_op.shape[1] - p1_op.shape[1])))

def pruner(I_parent, prune_percent, parent_key, children_key, children, clusters, labels, final_weights, model, testloader, device):

    # Create a copy
    init_weights   = final_weights.copy()

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
            if 'group' in alg:
                for group_1 in range(clusters):
                    if I_parent[str(num_layers)][child, group_1] < cutoff_value:
                        for group in np.where(labels[str(num_layers)]==group_1)[0]:
                            init_weights[children_k][child, group] = 0.

                    # END IF

                # END FOR

            else:
                for parent_1 in range(p1_op.shape[1]):
                    if I_parent[str(num_layers)][child, parent_1] < cutoff_value:
                        init_weights[children_k][child, parent_1] = 0.

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



    # Obtain Activations
    print("----------------------------------")
    print("Collecting activations from layers")

    act_start_time = time.time()
    p1_op = {}
    c1_op = {}
    # Conv1 and Conv2
    p1_op['0'], c1_op['0'] = activations(trainloader, model, device, parent_conv1=True, parent_conv2=False, parent_conv3=False, parent_conv4=False, parent_conv5=False, parent_linear1=False, parent_linear2=False, parent_linear3=False, child_conv1=False, child_conv2=True, child_conv3=False, child_conv4=False, child_conv5=False, child_linear1=False, child_linear2=False, child_linear3=False)
    # Conv2 and Conv3
    p1_op['1'], c1_op['1'] = activations(trainloader, model, device, parent_conv1=False, parent_conv2=True, parent_conv3=False, parent_conv4=False, parent_conv5=False, parent_linear1=False, parent_linear2=False, parent_linear3=False, child_conv1=False, child_conv2=False, child_conv3=True, child_conv4=False, child_conv5=False, child_linear1=False, child_linear2=False, child_linear3=False)
    # Conv3 and Conv4
    p1_op['2'], c1_op['2'] = activations(trainloader, model, device, parent_conv1=False, parent_conv2=False, parent_conv3=True, parent_conv4=False, parent_conv5=False, parent_linear1=False, parent_linear2=False, parent_linear3=False, child_conv1=False, child_conv2=False, child_conv3=False, child_conv4=True, child_conv5=False, child_linear1=False, child_linear2=False, child_linear3=False)
    # Conv4 and Conv5
    p1_op['3'], c1_op['3'] = activations(trainloader, model, device, parent_conv1=False, parent_conv2=False, parent_conv3=False, parent_conv4=True, parent_conv5=False, parent_linear1=False, parent_linear2=False, parent_linear3=False, child_conv1=False, child_conv2=False, child_conv3=False, child_conv4=False, child_conv5=True, child_linear1=False, child_linear2=False, child_linear3=False)
    # Conv5 and Linear1
    p1_op['4'], c1_op['4'] = activations(trainloader, model, device, parent_conv1=False, parent_conv2=False, parent_conv3=False, parent_conv4=False, parent_conv5=True, parent_linear1=False, parent_linear2=False, parent_linear3=False, child_conv1=False, child_conv2=False, child_conv3=False, child_conv4=False, child_conv5=False, child_linear1=True, child_linear2=False, child_linear3=False)
    # Linear1 and Linear2
    p1_op['5'], c1_op['5'] = activations(trainloader, model, device, parent_conv1=False, parent_conv2=False, parent_conv3=False, parent_conv4=False, parent_conv5=False, parent_linear1=True, parent_linear2=False, parent_linear3=False, child_conv1=False, child_conv2=False, child_conv3=False, child_conv4=False, child_conv5=False, child_linear1=False, child_linear2=True, child_linear3=False)
    # Linear2 and Linear3
    p1_op['6'], c1_op['6'] = activations(trainloader, model, device, parent_conv1=False, parent_conv2=False, parent_conv3=False, parent_conv4=False, parent_conv5=False, parent_linear1=False, parent_linear2=True, parent_linear3=False, child_conv1=False, child_conv2=False, child_conv3=False, child_conv4=False, child_conv5=False, child_linear1=False, child_linear2=False, child_linear3=True)
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

        if 'group' in alg:
            I_parent[str(idx)] = np.zeros((init_weights[children_key[idx]].shape[0], clusters))

        else:
            I_parent[str(idx)] = np.zeros((init_weights[children_key[idx]].shape[0], init_weights[children_key[idx]].shape[1]))

        # END IF

    # END FOR        
    #import pdb; pdb.set_trace()
 
    #labels       = np.zeros((nlayers,64))
    #children     = [192, 10]
    #I_parent     = np.zeros((nlayers, 192, p1_op.shape[1]))

    #if len(c1_op.shape) > 2:
    #    p1_op = p1_op.reshape(p1_op.shape[0],p1_op.shape[1],-1)
    #    c1_op = c1_op.reshape(c1_op.shape[0],c1_op.shape[1],-1)


    if alg == '1a':
        alg1a(nlayers, children, I_parent, p1_op, p2_op, c1_op, c2_op)

    elif alg == '1b':
        alg1b(nlayers, children, I_parent, p1_op, c1_op)

    else:
        alg1a_group(nlayers, children, I_parent, p1_op, c1_op, labels, clusters)

    # END IF

    print("----------------------------------")
    print("Begin Pruning of Weights")

    perf         = None
    prune_per    = None

    for prune_percent in np.arange(0.0, 1.0, step=0.1):
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
    parent_key   = ['conv1.weight','conv2.weight','conv3.weight','conv4.weight','conv5.weight',  'linear1.weight','linear2.weight']
    children_key = ['conv2.weight','conv3.weight','conv4.weight','conv5.weight','linear1.weight','linear2.weight','linear3.weight']
    alg          = '1a_group'
    clusters     = 8 

    perf, prune_per = calc_perf(parent_key, children_key, alg, clusters)

    plt.plot(prune_per, perf) 
    plt.xlabel('Ratio of weights pruned')
    plt.ylabel('Performance of AlexNet')
    plt.title('Comparison of performance variation when AlexNet is one-shot pruned, w/o retraining')
    plt.show()
