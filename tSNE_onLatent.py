# -*- coding: utf-8 -*-
"""
Basic t-SNE clustering to visualize mapping of 200-D latent space <-> 2D t-SNE space

"""

import numpy as np
import pickle
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import time

# =============================================================================
# Parameters
# =============================================================================
#path_to_zvectors = r"C:\Users\Grey\Desktop\3DGAN_shah\tsne_example\save-1-2_16K_zvectors.npy" #For now just open single file, eventually maybe multiple
#Xpath = r"C:\Users\Grey\Desktop\3DGAN_shah\tsne_example\save-1-2_16K.p"
#batchsize = 32

#[14,21,3] look similar
#modelnums = [14,21,3]+[5,15] +[7]+[18]+[9]#which of the 3d models to visualize (up to int = batchsize) (32)




#WHen using full set of 480 random + 32 interpolated
path_to_zvectors = "Z_out_for_tsne.npy"
Xpath = "X_out_for_tsne.npy"#Now that using numpy array instead. Path to the 3D space coordinates of the ~512 points

use_Z_or_X = 'X'#'X'#'Z'


# =============================================================================
# LOAD DATA
# =============================================================================
#For now, to create exampe figure. just load single bacth of inputs
Z = np.load(path_to_zvectors)
#print(Z)
#print(Z.shape) #Is [batchsize x 200]

#For validation only: what do the all zeros, all ones vectors look like in t-SNE:
#Z = np.vstack((Z,np.ones((1,Z.shape[1])),np.zeros((1,Z.shape[1])),-np.ones((1,Z.shape[1]))))
#Z = Z[:4]#smaller faster testing


X = np.load(Xpath)
X = X>.5
batchsize = X.shape[0]




#!!!!!!!!!
#test tsne on big daat
#Z = np.random.normal(size=(480,200))#64**3))




# =============================================================================
# Run t-SNE
# =============================================================================
np.random.seed(0)
print('Doing t-SNE...')
t1 = time.clock()
tsne = TSNE(n_components=2,
            perplexity=30.0, #!!!!!!!!!!!!! this parameter needs to be tuned to this problem
            early_exaggeration=12.0,
            learning_rate=200.0,
            n_iter=1000,
            n_iter_without_progress=300,
            min_grad_norm=1e-07,
            metric='euclidean',
#            init='pca',#'random',
            init='random',
            verbose=0,
            random_state=None,
            method='barnes_hut',
            angle=0.5)
if use_Z_or_X=='X':
    Y = tsne.fit_transform(X.reshape(X.shape[0],-1))
elif use_Z_or_X=='Z':
    Y = tsne.fit_transform(Z)
t2 = time.clock()
print('finished t-SNE transform')
print('Time: ',t2-t1)
print(Y)
print(Y.shape) #Is [batchsize x 200]


#Can also try multi-dimensional scaling as comparison
#mds = MDS()
#Y = mds.fit_transform(Z)




# =============================================================================
# For comparison, look at some 3D models
# =============================================================================
#with open(Xpath,'rb') as gg:
#    X = pickle.load(gg,encoding='latin1')#since my laptop is py3 and we pickled them in py2, encoding is different





    

    

#Plot t-SNE:
plt.figure()
if use_Z_or_X=='Z':
    plt.title('t-SNE Clustering of 200-D Latent Space',fontsize=20)
elif use_Z_or_X=='X':
    plt.title('t-SNE Clustering of 64^3 Space',fontsize=20)
    
    
plt.scatter(Y[:,0],Y[:,1])
axes = plt.gca()
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
#overlay labels to identify to match with 3D model:
ids = np.arange(Y.shape[0])
for i in range(len(ids)):
    x = Y[i,0]
    y = Y[i,1]
    #Only plot those interpolated indices:
    if i <= 33:
        plt.text(Y[i,0],Y[i,1],ids[i],fontsize=15)

    #Want to inset 3d axes in this one, but not working, so for now just do separately
    """
    #for some of the models, plot the 3D models overelaid    
    if i in modelnums:
        voxels = np.squeeze(X[i]>0.5)
        fig2 = plt.figure()
        a = plt.axes([x,y,10,20], facecolor='k')
        plt.title(str(i),fontsize=20)
        ax = fig2.gca(projection='3d')
        ax.voxels(voxels, facecolors='r', edgecolor='k')
        plt.xticks([])
        plt.yticks([])
    """

q=qqqqqq
#For now, just plot these separately and manually overlay
for i in modelnums:    
    voxels = np.squeeze(X[i]>0.5)
    #3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='r', edgecolor='k')
    plt.title('id='+str(i),fontsize=20)
    plt.show()
