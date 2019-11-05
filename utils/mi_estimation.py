import time
import copy
import sklearn

import numpy             as np
import scipy.spatial     as ss
import statsmodels.api   as sm
import matplotlib.pyplot as plt

from math                 import log, sqrt
from scipy                import stats
from scipy.sparse         import csr_matrix
from scipy.special        import digamma
from sklearn.neighbors    import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.feature_selection import mutual_info_regression as MI_class 

def NearestNeighborBootstrap(X_in, Y_in, Z_in, train_len=-1, k=1):  #Z_in is n by d-2 
    assert (type(X_in) == np.ndarray), "Not an array"
    assert (type(Y_in) == np.ndarray), "Not an array"
    assert (type(Z_in) == np.ndarray), "Not an array"
   

    nx, dx = X_in.shape
    ny, dy = Y_in.shape
    nz, dz = Z_in.shape

    assert (nx == ny), "Dimension Mismatch"
    assert (nz == ny), "Dimension Mismatch"
    assert (nx == nz), "Dimension Mismatch"
 
    samples = np.hstack([X_in, Y_in, Z_in])

    Xset = range(0, dx)
    Yset = range(dx, dx + dy)    
    Zset = range(dx + dy, dx + dy + dz)

    if train_len == -1:
        train_len = 2 * len(X_in) // 3

    assert (train_len < nx), "Training length cannot be larger than total length"

    train = samples[0:train_len, :]
    train_1 = samples[0:train_len//2, :]
    train_2 = samples[train_len//2:train_len, :]
    X = train_1[:, Xset]
    Y = train_1[:, Yset]
    Z = train_1[:, Zset]
    Yprime = train_2[:, Yset]
    nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree', metric='l2').fit(Z)
    distances, indices = nbrs.kneighbors(train_2[:, Zset])
    for i in range(len(train_1)):
        index = indices[i]
        Y[i, :] = Yprime[index, :]

    train = np.hstack([X, Y, Z])
    train1 = samples[train_len:, :]

    return train1, train


def dist(a, b):
    return np.linalg.norm(a-b)

def MSTgenerator(train, n):
    distanceMatrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            distanceMatrix[i, j] = dist(train[i], train[j])
    tree = minimum_spanning_tree(csr_matrix(distanceMatrix))
    return tree


def mi(X, Y, Z):
    n = X.shape[0]
    #n = 3 * 500
    n23 = 2 * n // 3
    n1 = n - n23
    I = []
    #MC_i = []
    iter = 1
    m_ = 0.
    for _ in range(iter):            
        train1, train23 = NearestNeighborBootstrap(X, Y, Z)
        train = np.append(train1, train23, axis=0)
        Tree = MSTgenerator(train, train.shape[0])
        #print("tree=", Tree)
        r = 0.
        for i in range(train.shape[0] - 1):
                if (Tree.indices[i] >= n1 and i < n1) or (Tree.indices[i] < n1 and i >= n1):
                    r += 1
        #print("r=",r)
        r = min(r, n1)
        #m_ = m_ + 1 - r * train.shape[0] / 2.0 / (n23//2) / n1
        m_ = m_ + 1 -  r / (n / 3)
       #mc = mc + MonteCarol(X, Y, Z, alpha,n)

    return m_/iter



 

if __name__=="__main__":
    start = time.time()

    n_samples   = 50

    data_matrix = np.random.multivariate_normal(mean=np.zeros((6,)), cov=np.identity(6), size=(n_samples))

    mid = time.time()

    x =data_matrix[:,:2].reshape(n_samples,2) 
    y =data_matrix[:,2:4].reshape(n_samples,2) 
    z =data_matrix[::,4:].reshape(n_samples,2)

    
    I_value = max(0,mi(x,y,z))

    end = time.time()
    print("Value is: %f", I_value)
    print("Time taken to generate data is : %f", mid - start)
    print("Time taken to calculate MI using knn is : %f", end - mid)
