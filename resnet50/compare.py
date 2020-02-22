import torch
import numpy as np

import os
import sys

from data_handler            import data_loader
from utils                   import activations, sub_sample_uniform, mi
from utils                   import save_checkpoint, load_checkpoint, accuracy, mi

from model                   import resnet50    as resnet50

if __name__=="__main__":

    orig_file = 'BASELINE_IMAGENET2012_RESNET50/0/logits_best.pkl'
    #mint_files = ['BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_39.96294556090583.pkl', 'BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_43.142143585397015.pkl','BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_47.683321288380064.pkl','BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_53.2144243945449.pkl',]
    #mint_files = ['BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_47.683321288380064.pkl','BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_53.2144243945449.pkl',]
    # Michigan Cluster
    mint_files = ['BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_46.0200087829206.pkl','BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_46.36413446262447.pkl','BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_46.69390344519785.pkl', 'BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_39.96294556090583.pkl', 'BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_43.142143585397015.pkl','BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_47.683321288380064.pkl','BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_53.2144243945449.pkl']
    #mint_files = ['BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_53.2144243945449.pkl']
    
    #### Load Data ####
    trainloader, testloader, extraloader = data_loader('IMAGENET', 64)

    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Network
    model = resnet50(num_classes=1000).to(device)

    ## Accuracy
    #init_weights = load_checkpoint(orig_file)
    #model.load_state_dict(init_weights)
    #acc = 100.*accuracy(model, testloader, device)
    #print('Accuracy of the original network: %f %%\n' %(acc))


    for mint_file in mint_files:
        # Accuracy 
        init_weights = load_checkpoint(mint_file)
        change_if_nec = {}
        for ite in init_weights.keys():
            if 'module' in ite:
                change_if_nec[ite.split('module.')[1]] = init_weights[ite]
            else:
                break
        if len(change_if_nec.keys()) > 0:
            model.load_state_dict(change_if_nec)
        else:
            model.load_state_dict(init_weights)
        acc = 100.*accuracy(model, testloader, device)
        print('Accuracy of MINT network with name %s is : %f %%\n' %(os.path.split(mint_file)[1],acc))
