""" Simple Code To Combine All Parallely Generated MI Estimates """

import argparse

import numpy as np

####  Combine and Save MI and Cluster Assignments ####
def combine(directory, prefix, load_keys):
    res_I_dict               = {}
    res_Labels_dict          = {}
    res_Labels_children_dict = {}
    
    for looper in range(len(load_keys)):
        res_I_dict[str(looper)]               = np.load(directory+'I_parent_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']
        res_Labels_dict[str(looper)]          = np.load(directory+'Labels_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']
        res_Labels_children_dict[str(looper)] = np.load(directory+'Labels_children_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']

    # END FOR
    
    np.save(directory+'I_parent_'+prefix+'.npy', res_I_dict)
    np.save(directory+'Labels_'+prefix+'.npy', res_Labels_dict)
    np.save(directory+'Labels_children_'+prefix+'.npy', res_Labels_children_dict)


if __name__=='__main__':

    """    
    Sample Inputs
    directory = 'results/MNIST_MLP_BATCH/0/'
    prefix = '10g'
    model  = 'mlp'

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--directory', type=str)
    parser.add_argument('--prefix',    type=str)
    parser.add_argument('--model',     type=str)

    args = parser.parse_args()

    load_keys  = ['_fc1.weight_fc2.weight', '_fc2.weight_fc3.weight']

    combine(args.directory, args.prefix, load_keys)
