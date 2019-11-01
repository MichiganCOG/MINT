import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_regression as MI_class 
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import time
import statsmodels.api as sm

 
def mi(x, y, z):
    """ Mutual information of x and y (conditioned on z if z is not None)
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """

    f_xz  = sm.nonparametric.KDEMultivariateConditional(endog=z, exog=x, dep_type='c'*z.shape[1], indep_type='c'*x.shape[1], bw='normal_reference')
    f_yz  = sm.nonparametric.KDEMultivariateConditional(endog=z, exog=y, dep_type='c'*z.shape[1], indep_type='c'*y.shape[1], bw='normal_reference')
    f_xyz = sm.nonparametric.KDEMultivariateConditional(exog=np.hstack((x,y)), endog=z, indep_type='c'*np.hstack((x,y)).shape[1], dep_type='c'*z.shape[1], bw='normal_reference')

    import pdb; pdb.set_trace()
    fair_idxs = np.where(f_xz.bw!=0.)[0]
    return 1. - 2.*np.mean(np.divide(np.multiply(f_xz.pdf(z[:,fair_idxs],x), f_yz.pdf(z[:, fair_idxs],y)), (np.multiply(f_xz.pdf(z[:, fair_idxs],x), f_yz.pdf(z[:, fair_idxs],y)) + f_xyz.pdf(z[:, fair_idxs], np.hstack((x,y))))))

    

def knn_mi(first_vector, second_vector, class_attrb):
    return max(0, mi(first_vector, second_vector, class_attrb))

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
