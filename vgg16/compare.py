import os
import sys
import copy
import torch
import scipy.sparse

import numpy               as np
import torch.nn            as nn
import torch.nn.functional as F

# Custom Imports
from pathlib                 import Path
from torch.optim             import SGD
from data_handler            import data_loader
from utils                   import activations, sub_sample_uniform, mi
from utils                   import save_checkpoint, load_checkpoint, accuracy, mi
from scipy.sparse            import csr_matrix
from PIL                     import Image

from model                   import VGG16_bn    as vgg
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

""" 
    Adversarial Attacks Section
    1. Iterative LL 
    2. Iterative FGSM

"""
def ll_test(model, device, test_loader, epsilon, steps ):

    model.eval()

    # Accuracy counter
    correct = 0
    total   = 0

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        total += data.shape[0]
        _, wrong_pred = torch.min(model(data).data, 1)
       
        perturbed_images = Variable(data, requires_grad=True)
 
        for idx in range(steps):
            zero_gradients(perturbed_images)
            output = model(perturbed_images)

            cost = nn.CrossEntropyLoss()(output, wrong_pred)
            cost.backward()
    
            perturbed_images.data =  perturbed_images.data - epsilon*perturbed_images.grad.sign()
            perturbed_images.data = torch.clamp(perturbed_images, torch.min(data).item()-epsilon, torch.max(data).item() + epsilon)



        # Re-classify the perturbed image
        output = model(perturbed_images)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += (final_pred[:,0] == target).sum().item()

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))

    return final_acc


def fgsm_test( model, device, test_loader, epsilon, steps):

    model.eval()
    # Accuracy counter
    correct = 0
    total   = 0

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        orig_pred = model(data).max(1)[1]
        total += data.shape[0]
       
        perturbed_images = Variable(data, requires_grad=True)
 
        for idx in range(steps):
            zero_gradients(perturbed_images)
            output = model(perturbed_images)

            cost = nn.CrossEntropyLoss()(output, target)
            cost.backward()
    
            perturbed_images.data = epsilon*perturbed_images.grad.sign() + perturbed_images.data
            perturbed_images.data = torch.clamp(perturbed_images, torch.min(data).item()-epsilon, torch.max(data).item() + epsilon)

        # Re-classify the perturbed image
        output = model(perturbed_images)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += (final_pred[:,0] == target).sum().item()

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(total)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))

    return final_acc



def compare_adversarial(results):

    #### Load Data ####
    trainloader, testloader, extraloader = data_loader('CIFAR10', 64)

    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = vgg(num_classes=10).to(device)

    print("Running Adversarial Attacks")

    for key in results.keys():
        state_dict = torch.load(key)['state_dict']
        model.load_state_dict(state_dict)

        # Iterative FGSM attack
        res_matrix = {}
        for epsilon in np.arange(0,0.01, 0.001):
            res_matrix[str(epsilon)] = fgsm_test(model, device, testloader, epsilon, 5)

        results[key]["fgsm"] = res_matrix 

        # Iterative LL attack
        res_matrix = {}
        for epsilon in np.arange(0,0.01, 0.001):
            res_matrix[str(epsilon)] = ll_test(model, device, testloader, epsilon, 5)

        results[key]["ll"] = res_matrix
 

def compare_accuracy(results):

    #### Load Data ####
    trainloader, testloader, extraloader = data_loader('CIFAR10', 64)

    # Check if GPU is available (CUDA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = vgg(num_classes=10).to(device)

    print("Running Accuracy Comparison")

    for key in results.keys():
        state_dict = torch.load(key)['state_dict']
        model.load_state_dict(state_dict)
        results[key]["accuracy"]= 100.*accuracy(model, testloader, device)

        # Temp to gen CSV file for calibration    
        with torch.no_grad():
            import csv
            with open(os.path.split(key)[1]+'.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs      = model(images, labels=True)
                    values, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).cpu().numpy()
                    for ite in np.vstack((values.cpu().numpy(), correct.astype('int32'))).T:
                        spamwriter.writerow(ite.tolist())

    print('-----------------------------------')
    print('Accuracy Stats For MLP on MNIST')
    print('-----------------------------------')
    for key in results.keys():
        print('File name: %s'%(os.path.split(key)[1]))
        print('Accuracy: %f'%(results[key]["accuracy"]))
        print('-----------------------------------')
    print('\n')


def compare_compression(results):

    print("Running Compression Comparison")
    orig_state_dict = None

    for key in sorted(results.keys()):
        state_dict = torch.load(key)['state_dict']

        if 'best' in key:
            orig_state_dict = torch.load(key)['state_dict']
            results[key]["compression"] = 0.0
            file_size = 0

            for key_dict in orig_state_dict.keys():
                if 'bn' in key_dict:
                    continue

                if len(orig_state_dict[key_dict].cpu().shape) > 2:
                    last_dim  = np.prod(orig_state_dict[key_dict].cpu().shape[1:])
                    first_dim = np.prod(orig_state_dict[key_dict].cpu().shape[0])
                    scipy.sparse.save_npz(key_dict+'.npz', csr_matrix(orig_state_dict[key_dict].cpu().numpy().reshape(first_dim, last_dim)))

                else:
                    scipy.sparse.save_npz(key_dict+'.npz', csr_matrix(orig_state_dict[key_dict].cpu()))

                file_size += Path(key_dict+'.npz').stat().st_size/(1024*1024.)

            results[key]["memory"] = file_size

        else:
            assert(len(orig_state_dict.keys()) == len(state_dict.keys()))
            

            total_params = 0
            exist_params = 0
            file_size 	 = 0


            for key_dict in orig_state_dict.keys():
                total_params = 0
                exist_params = 0

                if 'bn' in key_dict:
                    continue
                non_zero_params = len(np.where(state_dict[key_dict].reshape(-1).cpu()!=0)[0])

                exist_params += non_zero_params 
                total_params += orig_state_dict[key_dict].reshape(-1).shape[0]

                # Save orig and compressed version in CSR format for memory footprint
                if len(state_dict[key_dict].cpu().shape) > 2:
                    last_dim  = np.prod(state_dict[key_dict].cpu().shape[1:])
                    first_dim = np.prod(state_dict[key_dict].cpu().shape[0])
                    scipy.sparse.save_npz(key_dict+'.npz', csr_matrix(state_dict[key_dict].cpu().numpy().reshape(first_dim, last_dim)))

                else:
                    scipy.sparse.save_npz(key_dict+'.npz', csr_matrix(state_dict[key_dict].cpu()))

                file_size += Path(key_dict+'.npz').stat().st_size/(1024*1024.)
                print('Compression of layer %s is %f'%(key_dict, 1 - (exist_params/float(total_params))))


            results[key]["compression"] = 1 - (exist_params/float(total_params))
            results[key]["memory"]      = file_size

    print('-----------------------------------')
    print('Compression Stats For MLP on MNIST')
    print('-----------------------------------')
    for key in results.keys():
        print('File name: %s'%(os.path.split(key)[1]))
        print('Compression: %f'%(results[key]["compression"]*100.))
        print('Memory Footprint: %f Mb'%(results[key]["memory"]))
        print('-----------------------------------')
    print('\n')


if __name__=="__main__":

    results = {}
    orig_file       = 'results/BASELINE_CIFAR10_VGG16_BN/0/logits_best.pkl'

    results[orig_file] = {"fgsm": 0.0, "ll": 0.0, "accuracy": 0.0, "compression": 0.0}

    for compressed_file in ['results/BASELINE_CIFAR10_VGG16_BN_RETRAIN_1/0/logits_15.03558254454343.pkl']:
        results[compressed_file] = {"fgsm": 0.0, "ll": 0.0, "accuracy": 0.0, "compression": 0.0}

    # Run Comparisons
    #compare_class_viz(results)
    compare_adversarial(results)
    #compare_accuracy(results)
    #compare_compression(results)
    import pdb; pdb.set_trace()
