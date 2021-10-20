import torch
import copy
import random 

import numpy    as np
import torch.nn as nn


visualisation = {}

"""
Fwd Hook Code Acknowledgement: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
"""

#### Hook Function ####
def hook_fn(m, i, o):
    visualisation[m] = o 


#### Return Forward Hooks To All Layers ####
def get_all_layers(net, hook_handles, item_key):
    for name, layer in net._modules.items():
        if name == item_key.split('.')[0]:
            if isinstance(layer, nn.Sequential):
                get_all_layers(layer)

            else:
                hook_handles.append(layer.register_forward_hook(hook_fn))

            # END IF

        # END IF

    # END FOR

#### Return Activations From Desired Layers ####
def activations(data_loader, model, device, item_key):
    temp_op       = None
    temp_label_op = None

    parents_op  = None
    labels_op   = None

    if not(item_key == 'input'):
        handles     = []

        get_all_layers(model, handles, item_key)

    print('Collecting Activations for Layer %s'%(item_key))

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data
            model(x_input.to(device))

            if temp_op is None:
                if not(item_key == 'input'):
                    temp_op  = visualisation[list(visualisation.keys())[0]].cpu().numpy()

                else:
                    temp_op  = x_input.cpu().numpy()

                # END IF

                temp_labels_op = y_label.numpy()

            else:
                if not(item_key == 'input'):
                    temp_op = np.vstack((visualisation[list(visualisation.keys())[0]].cpu().numpy(), temp_op))
            
                else:
                    temp_op = np.vstack((x_input.cpu().numpy(), temp_op))

                # END IF

                temp_labels_op = np.hstack((y_label.numpy(), temp_labels_op))

            # END IF 

            if step % 100 == 0:
                if parents_op is None:
                    parents_op = copy.deepcopy(temp_op)
                    labels_op  = copy.deepcopy(temp_labels_op)

                    temp_op        = None
                    temp_labels_op = None

                else:
                    parents_op = np.vstack((temp_op, parents_op))
                    labels_op  = np.hstack((temp_labels_op, labels_op))

                    temp_op        = None
                    temp_labels_op = None

                # END IF

        # END FOR

    # END WITH 

    if parents_op is None:
        parents_op = copy.deepcopy(temp_op)
        labels_op  = copy.deepcopy(temp_labels_op)

        temp_op        = None
        temp_labels_op = None

    else:
        parents_op = np.vstack((temp_op, parents_op))
        labels_op  = np.hstack((temp_labels_op, labels_op))

        temp_op        = None
        temp_labels_op = None

    # END IF

    # Remove all hook handles
    if not(item_key == 'input'):
        for handle in handles:
            handle.remove()    

        # END FOR
    
        del visualisation[list(visualisation.keys())[0]]

    # END IF

    if len(parents_op.shape) > 2:
        if item_key == 'input':
            parents_op  = np.mean(parents_op, axis=(1)).reshape(parents_op.shape[0],-1)

        else:
            parents_op  = np.mean(parents_op, axis=(2,3))

        # END IF

    # END IF

    return parents_op, labels_op


#### Sub-sample and return activations, uniform prior over labels####
def sub_sample_uniform(activations, labels, num_samples_per_class=250):

    chosen_sample_idxs = []

    # Basic Implementation of Nearest Mean Classifier
    unique_labels = np.unique(labels)

    for idxs in range(len(unique_labels)):
        chosen_idxs     = np.random.choice(np.where(labels==unique_labels[idxs])[0],num_samples_per_class)
        chosen_sample_idxs.extend(chosen_idxs)

    # END FOR

    random.shuffle(chosen_sample_idxs)

    return activations[chosen_sample_idxs]
