import argparse

import numpy as np



def combine(directory, prefix, load_keys):
    res_I_dict               = {}
    res_Labels_dict          = {}
    res_Labels_children_dict = {}
    
    for looper in range(len(load_keys)):
        res_I_dict[str(looper)]               = np.load(directory+'I_parent_'+prefix+load_keys[looper]+'.npy').item()['0']
        res_Labels_dict[str(looper)]          = np.load(directory+'Labels_'+prefix+load_keys[looper]+'.npy').item()['0']
        res_Labels_children_dict[str(looper)] = np.load(directory+'Labels_children_'+prefix+load_keys[looper]+'.npy').item()['0']
    
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

    args = parser.parse_args()

    if args.model == 'mlp':
        load_keys  = ['_fc1.weight_fc2.weight', '_fc2.weight_fc3.weight']

    elif args.model == 'vgg':
        #load_keys  = ['_conv1.weight_conv2.weight','_conv2.weight_conv3.weight','_conv3.weight_conv4.weight','_conv4.weight_conv5.weight','_conv5.weight_conv6.weight','_conv6.weight_conv7.weight','_conv7.weight_conv8.weight','_conv8.weight_conv9.weight','_conv9.weight_conv10.weight', '_conv10.weight_conv11.weight','_conv11.weight_conv12.weight','_conv12.weight_conv13.weight','_conv13.weight_linear1.weight', '_linear1.weight_linear3.weight']
        load_keys  = ['_conv1.weight_conv2.weight','_conv8.weight_conv9.weight','_conv9.weight_conv10.weight', '_conv10.weight_conv11.weight','_conv11.weight_conv12.weight','_conv12.weight_conv13.weight','_conv13.weight_linear1.weight']

    combine(args.directory, args.prefix, load_keys)
