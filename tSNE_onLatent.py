# -*- coding: utf-8 -*-
"""
Basic t-SNE clustering to visualize mapping of 200-D latent space <-> 2D t-SNE space

"""

import numpy as np
import pickle
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Parameters
# =============================================================================
path_to_zvectors = r"C:\Users\Grey\Desktop\3DGAN_shah\tsne_example\save-1-2_16K_zvectors.npy" #For now just open single file, eventually maybe multiple
path_to_pickle = r"C:\Users\Grey\Desktop\3DGAN_shah\tsne_example\save-1-2_16K.p"
batchsize = 32

#[14,21,3] look similar
modelnums = [14,21,3]+[5,15] +[7]+[18]+[9]#which of the 3d models to visualize (up to int = batchsize) (32)










# =============================================================================
# LOAD DATA
# =============================================================================
#For now, to create exampe figure. just load single bacth of inputs
X = np.load(path_to_zvectors)
#print(X)
#print(X.shape) #Is [batchsize x 200]

#For validation only: what do the all zeros, all ones vectors look like in t-SNE:
#X = np.vstack((X,np.ones((1,X.shape[1])),np.zeros((1,X.shape[1])),-np.ones((1,X.shape[1]))))
#X = X[:4]#smaller faster testing


# =============================================================================
# Run t-SNE
# =============================================================================
np.random.seed(0)
tsne = TSNE(n_components=2,
            perplexity=5.0, #!!!!!!!!!!!!! this parameter needs to be tuned to this problem
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
Y = tsne.fit_transform(X)
print(Y)
print(Y.shape) #Is [batchsize x 200]


#Can also try multi-dimensional scaling as comparison
#mds = MDS()
#Y = mds.fit_transform(X)




# =============================================================================
# For comparison, look at some 3D models
# =============================================================================
with open(path_to_pickle,'rb') as gg:
    g_objects = pickle.load(gg,encoding='latin1')#since my laptop is py3 and we pickled them in py2, encoding is different
    batchsize = g_objects.shape[0]
    

    

#Plot t-SNE:
plt.figure()
plt.title('t-SNE Clustering of 200-D Latent Space',fontsize=20)
plt.scatter(Y[:,0],Y[:,1])
axes = plt.gca()
ymin, ymax = axes.get_ylim()
xmin, xmax = axes.get_xlim()
#overlay labels to identify to match with 3D model:
ids = np.arange(Y.shape[0])
for i in range(len(ids)):
    x = Y[i,0]
    y = Y[i,1]
    plt.text(Y[i,0],Y[i,1],ids[i])

    #Want to inset 3d axes in this one, but not working, so for now just do separately
    """
    #for some of the models, plot the 3D models overelaid    
    if i in modelnums:
        voxels = np.squeeze(g_objects[i]>0.5)
        fig2 = plt.figure()
        a = plt.axes([x,y,10,20], facecolor='k')
        plt.title(str(i),fontsize=20)
        ax = fig2.gca(projection='3d')
        ax.voxels(voxels, facecolors='r', edgecolor='k')
        plt.xticks([])
        plt.yticks([])
    """


#For now, just plot these separately and manually overlay
for i in modelnums:    
    voxels = np.squeeze(g_objects[i]>0.5)
    #3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='r', edgecolor='k')
    plt.title('id='+str(i),fontsize=20)
    plt.show()
