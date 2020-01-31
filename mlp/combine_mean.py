""" Simple Code To Combine All Parallely Generated MI Estimates """

import argparse

import numpy as np

def combine(directory, prefix, load_keys, trials):
    res_I_dict               = {}
    res_Labels_dict          = {}
    res_Labels_children_dict = {}
    
    for trial in range(trials):
        for looper in range(len(load_keys)):
            if trial == 0:
                res_I_dict[str(looper)]               = np.load(directory+'I_parent_'+'trial_'+str(int(trial+1))+'_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']
                res_Labels_dict[str(looper)]          = np.load(directory+'Labels_'+'trial_'+str(int(trial+1))+'_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']
                res_Labels_children_dict[str(looper)] = np.load(directory+'Labels_children_'+'trial_'+str(int(trial+1))+'_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']
            else:
                res_I_dict[str(looper)]               += np.load(directory+'I_parent_'+'trial_'+str(int(trial+1))+'_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']
                res_Labels_dict[str(looper)]          += np.load(directory+'Labels_'+'trial_'+str(int(trial+1))+'_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']
                res_Labels_children_dict[str(looper)] += np.load(directory+'Labels_children_'+'trial_'+str(int(trial+1))+'_'+prefix+load_keys[looper]+'.npy', allow_pickle=True).item()['0']
    
   
    for looper in range(len(load_keys)):
        res_I_dict[str(looper)]/=5.
        res_Labels_dict[str(looper)]          = (res_Labels_dict[str(looper)]/trials).astype('int32')
        res_Labels_children_dict[str(looper)] = (res_Labels_children_dict[str(looper)]/trials).astype('int32')
 
    np.save(directory+'I_parent_'+prefix+'.npy', res_I_dict)
    np.save(directory+'Labels_'+prefix+'.npy', res_Labels_dict)
    np.save(directory+'Labels_children_'+prefix+'.npy', res_Labels_children_dict)


if __name__=='__main__':

    """    
    Sample Inputs
    directory = '/z/home/madantrg/Pruning/results/MNIST_MLP_BATCH/0/'
    prefix = '10g'
    model  = 'mlp'

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--directory', type=str)
    parser.add_argument('--prefix',    type=str)
    parser.add_argument('--model',     type=str)
    parser.add_argument('--trials',    type=int)

    args = parser.parse_args()

    load_keys  = ['_fc1.weight_fc2.weight', '_fc2.weight_fc3.weight']

    combine(args.directory, args.prefix, load_keys, args.trials)
