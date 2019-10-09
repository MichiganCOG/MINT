import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_regression as MI_class 
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import time


#def entropy(counts):
#    '''Compute entropy.'''
#    ps = counts/float(np.sum(counts))  # coerce to float and normalize
#    #ps = ps[np.nonzero(ps)]            # toss out zeros
#    #H = -sum(ps * numpy.log2(ps))   # compute entropy
#    
#    return ps
#
#def entropy_label(counts):
#    '''Compute entropy.'''
#    ps = counts/float(np.sum(counts))  # coerce to float and normalize
#    ps = ps[np.nonzero(ps)]            # toss out zeros
#    #H = -sum(ps * numpy.log2(ps))   # compute entropy
#    
#    return ps
#
#
#def mi(x, y, z, binss=64):
#    counts_xyz = entropy(np.histogramdd(np.hstack((x, y, z)), bins=binss)[0])#, range=[[0, 311335], [0, 311335], [0, 311335]])[0])
#
#    counts_xy  = entropy(np.histogramdd(np.hstack((x, y)), bins=binss)[0])#, range=[[0, 311335], [0, 311335]])[0])
#    counts_yz  = entropy(np.histogramdd(np.hstack((y, z)), bins=binss)[0])#, range=[[0, 311335], [0, 311335]])[0])
#    
#    counts_z   = entropy(np.histogramdd(z, bins=binss)[0])#, range=[0, 311335])[0])
#
#    values = 0.0
#
#    for idx in range(binss):
#        import pdb; pdb.set_trace()
#        if ((counts_xy[idx] !=0) and (counts_yz[idx]!=0)):
#            values += counts_xyz[idx] * np.log(counts_xyz[idx]*counts[z]/(counts_xy[idx]*counts_yz[idx]))

#    return values 

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]

def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    n_elements = len(points)
    tree = ss.cKDTree(points)
    avg = 0.
    dvec = dvec - 1e-15
    for point, dist in zip(points, dvec):
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(point, dist, p=float('inf')))
        avg += digamma(num_points) / n_elements
    return avg
 
def mi(x, y, z=None, k=3, base=2):
    """ Mutual information of x and y (conditioned on z if z is not None)
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(yz, dvec), avgdigamma(z, dvec), digamma(k)
    return (-a - b + c + d) / log(base)
    


def knn_mi(first_vector, second_vector, class_attrb):
    first_vector  = first_vector.reshape(-1, 1)
    second_vector = second_vector.reshape(-1, 1)
    class_attrb   = class_attrb.reshape(-1, 1)

    return max(0, mi(first_vector, second_vector, class_attrb))

if __name__=="__main__":
    start = time.time()

    n_samples   = 5000

    data_matrix = np.random.multivariate_normal(mean=np.zeros((n_samples,)), cov=np.identity(n_samples), size=(3))

    mid = time.time()

    x =data_matrix[0,:].reshape(n_samples,1) 
    y =data_matrix[1,:].reshape(n_samples,1) 
    z =data_matrix[2,:].reshape(n_samples,1)

    
    I_value = max(0,mi(x,y,z))

    end = time.time()
    print("Value is: %f", I_value)
    print("Time taken to generate data is : %f", mid - start)
    print("Time taken to calculate MI using knn is : %f", end - mid)
