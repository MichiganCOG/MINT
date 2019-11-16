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

#### Activation function ####
def activations_vgg(data_loader, model, device, item_key):
    parents_op  = None
    labels_op   = None

    print('Collecting Activations for Layer %s\n'%(item_key))

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            x_input, y_label = data

            if parents_op is None:
                if 'conv1.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv1=True).cpu().numpy(), axis=(2,3))
                elif 'conv2.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv2=True).cpu().numpy(), axis=(2,3))
                elif 'conv3.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv3=True).cpu().numpy(), axis=(2,3))
                elif 'conv4.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv4=True).cpu().numpy(), axis=(2,3))
                elif 'conv5.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv5=True).cpu().numpy(), axis=(2,3))
                elif 'conv6.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv6=True).cpu().numpy(), axis=(2,3))
                elif 'conv7.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv7=True).cpu().numpy(), axis=(2,3))
                elif 'conv8.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv8=True).cpu().numpy(), axis=(2,3))
                elif 'conv9.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv9=True).cpu().numpy(), axis=(2,3))
                elif 'conv10.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv10=True).cpu().numpy(), axis=(2,3))
                elif 'conv11.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv11=True).cpu().numpy(), axis=(2,3))
                elif 'conv12.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv12=True).cpu().numpy(), axis=(2,3))
                elif 'conv13.' in item_key:
                    parents_op  = np.mean(model(x_input.to(device), conv13=True).cpu().numpy(), axis=(2,3))
                elif 'linear1.' in item_key:
                    parents_op  = model(x_input.to(device), linear1=True).cpu().numpy()
                elif 'linear3.' in item_key:
                    parents_op  = model(x_input.to(device)).cpu().numpy()

                labels_op = y_label.numpy()

            else:
                if 'conv1.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv1=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv2.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv2=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv3.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv3=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv4.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv4=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv5.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv5=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv6.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv6=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv7.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv7=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv8.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv8=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv9.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv9=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv10.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv10=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv11.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv11=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv12.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv12=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'conv13.' in item_key:
                    parents_op  = np.vstack((np.mean(model(x_input.to(device), conv13=True).cpu().numpy(), axis=(2,3)), parents_op))
                elif 'linear1.' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device), linear1=True).cpu().numpy(), parents_op))
                elif 'linear3.' in item_key:
                    parents_op  = np.vstack((model(x_input.to(device)).cpu().numpy(), parents_op))

                labels_op = np.hstack((labels_op, y_label.numpy()))

            # END IF 

        # END FOR

    # END FOR

    if len(parents_op.shape) > 2:
        parents_op  = np.mean(parents_op, axis=(2,3))

    return parents_op, labels_op

#### Sub-sample function ####
def sub_sample(activations, labels, num_samples_per_class=250):

    chosen_sample_idxs = []

    # Basic Implementation of Nearest Mean Classifier
    unique_labels = np.unique(labels)
    centroids     = np.zeros((len(unique_labels), activations.shape[1]))

    for idxs in range(len(unique_labels)):
        centroids[idxs] = np.mean(activations[np.where(labels==unique_labels[idxs])[0]], axis=0)
        chosen_idxs     = np.argsort(np.linalg.norm(activations[np.where(labels==unique_labels[idxs])[0]] - centroids[idxs], axis=1))[:num_samples_per_class]

        chosen_sample_idxs.extend((np.where(labels==unique_labels[idxs])[0])[chosen_idxs].tolist())

    
    return activations[chosen_sample_idxs]



