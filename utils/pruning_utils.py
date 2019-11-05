import torch
import copy

import numpy as np


#### Activation function ####
def activations_mlp(data_loader, model, device, item_key):
    parents_op  = None
    labels_op   = None

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data

            if parents_op is None:
                if 'fc1' in item_key:
                    parents_op  = model(x_input.to(device), fc1=True).cpu().numpy()
                elif 'fc2' in item_key:
                    parents_op  = model(x_input.to(device), fc2=True).cpu().numpy()
                elif 'fc3' in item_key:
                    parents_op  = model(x_input.to(device)).cpu().numpy()

                labels_op = y_label.numpy()

            else:
                if 'fc1' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), fc1=True).cpu().numpy(), parents_op))
                elif 'fc2' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), fc2=True).cpu().numpy(), parents_op))
                elif 'fc3' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device)).cpu().numpy(), parents_op))

                labels_op = np.hstack((labels_op, y_label.numpy()))

            # END IF 

        # END FOR

    # END FOR


    if len(parents_op.shape) > 2:
        parents_op  = np.mean(parents_op, axis=(2,3))

    return parents_op, labels_op


#### Sub-sample function ####
def sub_sample(activations, labels, num_samples_per_class=100):

    chosen_sample_idxs = []

    # Basic Implementation of Nearest Mean Classifier
    unique_labels = np.unique(labels)
    centroids     = np.zeros((len(unique_labels), activations.shape[1]))

    for idxs in range(len(unique_labels)):
        centroids[idxs] = np.mean(activations[np.where(labels==unique_labels[idxs])[0]], axis=0)
        chosen_idxs     = np.argsort(np.linalg.norm(activations[np.where(labels==unique_labels[idxs])[0]] - centroids[idxs], axis=1))[:num_samples_per_class]

        chosen_sample_idxs.extend((np.where(labels==unique_labels[idxs])[0])[chosen_idxs].tolist())

    
    return activations[chosen_sample_idxs]



