# -*- coding: utf-8 -*-
"""
Simple pairwise distances between points to use for numerical metrics
- Used for distance metrics in the high diemensional (64^3) space, as well as
  lower dimensional 200-D latent space.
"""


import numpy as np
from sklearn.metrics.pairwise import paired_distances, pairwise_distances
import matplotlib.pyplot as plt


#sklearn.metrics.pairwise.paired_distances(X, Y, metric='euclidean', **kwds)
#sklearn.metrics.pairwise.pairwise_distances(X, Y=None, metric='euclidean', n_jobs=1, **kwds)

def Get_Distances_L2(X):
    """
    Get the L2 distances between each of the Nsamples vectors.
    
    In the high dimensional space each vector is length 64^3.
    
    In the latent space each vector is length 200.
    
    If doing on t-SNE coordinates, each vector is length 2.
    """
    DistanceMatrix = pairwise_distances(X,metric='euclidean')
    #A distance matrix D such that D_{i, j} is the distance between the 
    #ith and jth vectors of the given matrix X, if Y is None. 
    return DistanceMatrix

def Get_Distances_Jaccard(X):
    """
    Get the Jaccard distances (1 - Intersection Over Union) between each of the Nsamples vectors.
    """
    DistanceMatrix = pairwise_distances(X,metric='jaccard')
    #A distance matrix D such that D_{i, j} is the distance between the 
    #ith and jth vectors of the given matrix X, if Y is None. 
    return DistanceMatrix


def Compare_Latent_to_3D(X,Z):
    """
    Check that points that are similar in 3D space [small Jaccard or L2],
    are also close in the latent space [small L2].
    Make basic plot to look for correlation.
    
    X - array of shape Nexamples x Ndims (e.g. 500 x 64^3)
    
    Z - array of shape Nexamples x Ndims (e.g. 500 x 200)
    """
    d_Z__L2 = Get_Distances_L2(Z)#.flatten()
#    d_X__L2 = Get_Distances_L2(X).flatten()
    d_X__Jaccard = Get_Distances_Jaccard(X)#.flatten()
    print('Done calculating distances')
    
    
    #Since there are some garbage generated results, 
    #for a given object, only look at the distance to the nearest K
    x = []
    z = []
    K = 2#5 nearest
    for pt in range(d_X__Jaccard.shape[0]):
        inds = np.argsort(d_X__Jaccard[pt])[:K]
        x += list(d_X__Jaccard[pt][inds])
        z += list(d_Z__L2[pt][inds])
    
    
    
    
    #Plot values for single object
#    d_Z__L2 = d_Z__L2[:480]
#    d_X__Jaccard = d_X__Jaccard[:480]
    
    
#    #To not plot diagonal terms [distance 0 from point to itself]
#    d_Z__L2 = d_Z__L2[d_Z__L2>0.]
##    d_X__L2 = d_X__L2[d_X__L2>0.]
#    d_X__Jaccard = d_X__Jaccard[d_X__Jaccard>0.]
    

    
    plt.figure()
    plt.title('High-D vs. Low-D Distance Correlations')
#    plt.plot(d_X__Jaccard.flatten(), d_Z__L2.flatten(), marker='o', linestyle='None')
#    plt.plot(d_X__L2.flatten(), d_Z__L2.flatten(), marker='o', linestyle='None')
    plt.plot(x,z, marker='o', linestyle='None')
    plt.xlabel('High-D Jaccard Distance',fontsize=20)
    plt.ylabel('Low-D L2 Distance',fontsize=20)
    plt.show()
    
    
    
    
    
#Y = np.random.randint(0,2,size=(500,64**3)).astype(bool)
#D = Get_Distances_Jaccard(Y) #Takes about 30 secs on laptop CPU
    
    
    
if __name__ == "__main__":
    Xpath = r"../chairs_480pts/chairs_15batches.npy"
    Zpath = r"../chairs_480pts/save-16K_15batches_zvectors.npy"
    X = np.load(Xpath)
    Z = np.load(Zpath)
    Npoints = X.shape[0]
    X = X.reshape(Npoints,64**3)
    #Similar to in plotting, to say occupied you need to threshold > T: (.9 or .5)
    X = X>0.9
    
    #just plot a few points for now:
    K=50;X=X[:K];Z=Z[:K]
    
    Compare_Latent_to_3D(X,Z)
    
    