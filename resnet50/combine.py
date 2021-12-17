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

    load_keys  = ['_conv1.weight_conv2.weight', '_conv2.weight_conv3.weight', '_conv3.weight_conv4.weight', '_conv4.weight_conv5.weight',  '_conv5.weight_conv6.weight',
                  '_conv6.weight_conv7.weight', '_conv7.weight_conv8.weight', '_conv8.weight_conv9.weight', '_conv9.weight_conv10.weight', '_conv10.weight_conv11.weight',
                  '_conv11.weight_conv12.weight', '_conv12.weight_conv13.weight', '_conv13.weight_conv14.weight', '_conv14.weight_conv15.weight',  '_conv15.weight_conv16.weight',
                  '_conv16.weight_conv17.weight', '_conv17.weight_conv18.weight', '_conv18.weight_conv19.weight', '_conv19.weight_conv20.weight', '_conv20.weight_conv21.weight',
                  '_conv21.weight_conv22.weight', '_conv22.weight_conv23.weight', '_conv23.weight_conv24.weight', '_conv24.weight_conv25.weight',  '_conv25.weight_conv26.weight',
                  '_conv26.weight_conv27.weight', '_conv27.weight_conv28.weight', '_conv28.weight_conv29.weight', '_conv29.weight_conv30.weight', '_conv30.weight_conv31.weight',
                  '_conv31.weight_conv32.weight', '_conv32.weight_conv33.weight', '_conv33.weight_conv34.weight', '_conv34.weight_conv35.weight',  '_conv35.weight_conv36.weight',
                  '_conv36.weight_conv37.weight', '_conv37.weight_conv38.weight', '_conv38.weight_conv39.weight', '_conv39.weight_conv40.weight', '_conv40.weight_conv41.weight',
                  '_conv41.weight_conv42.weight', '_conv42.weight_conv43.weight', '_conv43.weight_conv44.weight', '_conv44.weight_conv45.weight',  '_conv45.weight_conv46.weight',
                  '_conv46.weight_conv47.weight', '_conv47.weight_conv48.weight', '_conv48.weight_conv49.weight', '_conv49.weight_conv50.weight', '_conv50.weight_conv51.weight',
                  '_conv51.weight_conv52.weight', '_conv52.weight_conv53.weight']


    combine(args.directory, args.prefix, load_keys)
