#rearrange the X and Z matrices, since after doing interpolation we also want 
#to include those additional X (3D models) and Z (latent vectors) in our 
#t-SNE plots


import numpy as np




# =============================================================================
# PARAMETERS
# =============================================================================
#Indices of the 2 end points used for interpolation
#These correspond to the models that were manually chosen for interpolation end points:
z1_ind = 403
z2_ind = 59

#For the full set of ~480 points:
Xfullpath = r"../chairs_480pts/chairs_15batches.npy"
Zfullpath = r"../chairs_480pts/save-16K_15batches_zvectors.npy"

#For the ~32 or batchsize interpolated points:
X_interpolated_path = r"../chairs_480pts/save-16K_15batches__final__interpolated_results.npy"
Z_interpolated_path = r"chairs/interpolated_chairs_403_59.npy"

#Savenames to use for rerganized X and Z data, which will be put into t-SNE:
Xsavename = 'X_out_for_tsne.npy'
Zsavename = 'Z_out_for_tsne.npy'




# =============================================================================
# MAIN
# =============================================================================
#Load data
Xfull = np.load(Xfullpath)#(480, 64, 64, 64, 1)
Zfull = np.load(Zfullpath)#(480, 200)
Zinterpolated = np.load(Z_interpolated_path)#(32, 200)
Xinterpolated = np.load(X_interpolated_path)#(32, 64, 64, 64, 1)
#Correct the Xinterpolated
#(remember they were much noisier, so to try to make fair comparison, trim the boudnaries)
mask = np.ones(Xinterpolated[0].shape)
mask[:10] = 0.
mask[-20:] = 0.
mask[:,:20] = 0.
mask[:,-20:] = 0.
mask[:,:,-10:] = 0.
for m in range(Xinterpolated.shape[0]):
    Xinterpolated[m] *= mask



#Combine and reorganzie so that we can keep track of it on the tSNE plots
inds = [i for i in range(Xfull.shape[0]) if i not in [z1_ind,z2_ind]]
#print(inds)
#print(Xfull[z1_ind].shape)
X = np.concatenate((np.expand_dims(Xfull[z1_ind],axis=0), Xinterpolated, np.expand_dims(Xfull[z2_ind],axis=0), Xfull[inds]), axis=0)
print(X.shape)

#Same for Z:
Z = np.concatenate((np.expand_dims(Zfull[z1_ind],axis=0), Zinterpolated, np.expand_dims(Zfull[z2_ind],axis=0), Zfull[inds]), axis=0)
print(Z.shape)



np.save(Xsavename,X)
np.save(Zsavename,Z)
print('saved files  ', Xsavename, Zsavename)