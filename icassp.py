import torch
import copy
import torch

import numpy             as np
import matplotlib.pyplot as plt

from data_handler         import data_loader
from models               import Alexnet        as alex
from utils                import save_checkpoint, load_checkpoint, accuracy
from scipy.sparse         import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def dist(a, b):
    return np.linalg.norm(a-b)

def MSTgenerator(train, n):
    distanceMatrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            distanceMatrix[i, j] = dist(train[i], train[j])
    tree = minimum_spanning_tree(csr_matrix(distanceMatrix), overwrite=True)
    return tree

def MYDELTA(nd, nt, mst, N, m, s, l):
    
    # Split the data
    n = np.floor(N/2)
 
    # Compute the statistic
    if s==0:
        M = sum(np.multiply((mst[:,0] <= nd[0,0]), np.logical_and(mst[:,1] <= n + sum(nt[0, :l+1]) , mst[:,1] > n + sum(nt[0, :l])).astype('float32')))    +       sum(np.multiply((mst[:,1] <= nd[0,0]), np.logical_and(mst[:,0] >  n + sum(nt[0, :l+1])   , mst[:,0] <=n+sum(nt[0,:l])).astype('float32')))


    else:
        M = sum(np.multiply(np.logical_and(mst[:,0] > sum(nd[0,:s]) , mst[:,0] <= sum(nd[0, :s+1])), np.logical_and(mst[:,1] > n+sum(nt[0, :l]) , mst[:,1] <= n+sum(nt[0, :l+1])).astype('float32')))      +     sum(np.multiply(np.logical_and(mst[:,1] > sum(nd[0,:s]) , mst[:,1] <= sum(nd[0, :s+1])), np.logical_and(mst[:,0] > n+sum(nt[0, :l]) , mst[:,0] <= n+sum(nt[0, :l+1])).astype('float32')));

    myFR = (nd[0,s]/n)*(nt[0,l]/n)*(n/(2*nd[0,s]*nt[0,l]))*M;
    
    return myFR


def MYDATAR(data, col_idx, total_elements):
    data[:, 1] = data[np.random.choice(np.arange(total_elements,dtype=int), int(total_elements), replace='True'), 1]
    data[:, 2] = data[np.random.choice(np.arange(total_elements,dtype=int), int(total_elements), replace='False'), 2]
    
    return data

def split_data(data_matrix, data_size, total_classes):
    
    # Split the data into equal halves
    n  = np.floor(data_size/2).astype('int')
    X1 = data_matrix[:n, :]
    X2 = data_matrix[n:, :]

    # Sort the data
    XS1 = X1[np.argsort(X1[:,2]),:]
    XS2 = X2[np.argsort(X2[:,2]),:]

    # Split data by class label values
    nd = np.zeros((1, total_classes))
    nt = np.zeros((1, total_classes))

    for label_idx in range(total_classes):
        nd[0, label_idx] = len(np.where(XS1[:, 2]==label_idx)[0])
        nt[0, label_idx] = len(np.where(XS2[:, 2]==label_idx)[0])


    mst = 0.0

    # Skipping the mat2cell operation since we don't need it

    kd = 0
    kt = 0
 
    DTtempt0 = np.zeros((2, n)) 
    IDtempt1 = np.zeros((2, n)) 

    ID = {}

    for y in range(total_classes):
        IDtempt1[:, kt:kt+int(nt[0,y])] = MYDATAR(XS2[np.where(XS2[:,2]==y)[0], :], 3, nt[0, y])[:, :2].T
        kt        += int(nt[0, y])
        
        DTtempt0[:, kd:kd+int(nd[0,y])] = XS1[np.where(XS1[:,2]==y)[0], :2].T
        kd        += int(nd[0, y])

    mst_input = np.hstack((DTtempt0, IDtempt1)).T
    mst       = MSTgenerator(mst_input, mst_input.shape[0]).toarray()
    mst       = csr_matrix(mst).toarray()
    mst_edges = np.zeros((np.count_nonzero(mst), 2))

    mst_edges[:,0] = np.where(mst!=0.)[0]
    mst_edges[:,1] = np.where(mst!=0.)[1]

    return nd, nt, mst_edges 


def icassp(first_vector, second_vector, class_attrb):

    first_vector  = first_vector.reshape(-1,1)
    second_vector = second_vector.reshape(-1,1)
    class_attrb   = class_attrb.reshape(-1,1)

    total_classes = len(np.unique(class_attrb))
    mydelta       = np.zeros((1, total_classes, total_classes))

    data_matrix   = np.concatenate((first_vector, second_vector, class_attrb), axis=1)
    data_size     = first_vector.shape[0]

    [nd, nt, mst] = split_data(data_matrix, data_size, total_classes)


    for i_idx in range(total_classes):
        for j_idx in range(total_classes):
            mydelta[0, i_idx, j_idx] = MYDELTA(nd, nt, mst, data_size, total_classes, i_idx, j_idx)
    
    myCMI0 = np.sum(mydelta, 2)
    score  = np.maximum(0, 1-2*np.sum(myCMI0))   

    return score
