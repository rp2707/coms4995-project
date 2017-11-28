# -*- coding: utf-8 -*-
"""
Interpolating between vectors in 200-D latent space

Following these 2 references:
-https://arxiv.org/pdf/1609.04468.pdf
-https://github.com/dribnet/plat
we use spherical linear interpolation rather than simple linear interpolation,
which is said to improve results.
"""


import numpy as np




# =============================================================================
# Parameters
# =============================================================================
path_to_latent_vector_1 = r"111111.npy" #Path to the .npy file containing latent space point 1
path_to_latent_vector_2 = r"222222.npy" #Path to the .npy file containing latent space point 2

N_interpolations = 32 #Number of interpolation points to do between 1 and 2
#for now, to make easier to put in tensorflow, use N_interpolations=batch_size = 32,
#so can just rerun testGAN function on this batch of z vectors



# =============================================================================
# LOAD DATA
# =============================================================================
#Load the numpy array (.npy file) which has latent vector
z1 = np.load(path_to_latent_vector_1)
z2 = np.load(path_to_latent_vector_2)



# =============================================================================
# INTERPOLATION
# =============================================================================
#Do spherical interpolation following the listed references
#interpolations = ...


# =============================================================================
# SAVE
# =============================================================================
#Save out the interpolated vectors as numpy array
#has dimensions [batch_size, z_size]
np.save('interpolations.npy',interpolations)

#!!!!!!!!
#This file then needs to be fed into TestGAN in tensorflow code to make 3D models from these newly made z vectors
