# -*- coding: utf-8 -*-
"""
visualize output of GAN
- Use same >.5 thresholding on GAN output voxel probabilities
- Plot as binary occupancy voxel grid
- Could also look at actual probabilities in 3d space [darker is higher probability]
  since these values will eventually be adjusted with more training

"""

import numpy as np
import pickle

#import matplotlib
#matplotlib.use('Agg') #GTKAgg
import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D




# =============================================================================
# Parameters
# =============================================================================
#Using new network
#path_to_pickle = r"C:\Users\Grey\Desktop\3DGAN_shah\save-1-2-4200__newnet.p"
#path_to_pickle = r"C:\Users\Grey\Desktop\3DGAN_shah\save-1-2-8000_newnetwork.p"
#path_to_pickle = r"C:\Users\Grey\Desktop\3DGAN_shah\save-1-2-9600_newnetwork.p"
path_to_pickle = r"C:\Users\Grey\Desktop\3DGAN_shah\save-1-2-20K.p"
savename = '1-2-20K'

PlotHistograms = False
PlotAllModels = True
modelnum = 4 #which of the 3d models to visualize (up to int = batchsize) (32)
batchsize = 32


# =============================================================================
# Main
# =============================================================================
with open(path_to_pickle,'rb') as gg:
    #g_objects = pickle.load(gg)
    g_objects = pickle.load(gg,encoding='latin1')#since my laptop is py3 and we pickled them in py2, encoding is different
    batchsize = g_objects.shape[0]
    #print(g_objects)


for k in range(16):#range(batchsize):
    
    if not PlotAllModels and k!=modelnum:
        continue
        
    #np.squeeze(g_objects[id_ch[i]]>0.5)
    #k = 0 #[0,batchsize] #default code has batchsize 32
    #Get s ingle example of a voxel model for viewing
    voxels = np.squeeze(g_objects[k]>0.5)
    
    if PlotHistograms:
        plt.figure()
        plt.hist(g_objects[k].flatten(),bins=100)
        plt.show()
    
    
    #3D plot
    fig = plt.figure()
    plt.title(str(modelnum))
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='r', edgecolor='k')
#    plt.savefig("{0}_k{1}.png".format(savename,k))
