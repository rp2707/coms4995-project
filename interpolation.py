#From the repo at:
#https://github.com/dribnet/plat/blob/master/plat/interpolate.py


import numpy as np
from scipy.stats import norm

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

def lerp_gaussian(val, low, high):
    """Linear interpolation with gaussian CDF"""
    low_gau = norm.cdf(low)
    high_gau = norm.cdf(high)
    lerped_gau = lerp(val, low_gau, high_gau)
    return norm.ppf(lerped_gau)

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def slerp_gaussian(val, low, high):
    """Spherical interpolation with gaussian CDF (generally not useful)"""
    offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
    low_gau_shifted = norm.cdf(low) - offset
    high_gau_shifted = norm.cdf(high) - offset
    circle_lerped_gau = slerp(val, low_gau_shifted, high_gau_shifted)
    epsilon = 0.001
    clipped_sum = np.clip(circle_lerped_gau + offset, epsilon, 1.0 - epsilon)
    result = norm.ppf(clipped_sum)
    return result

def get_interpfn(spherical, gaussian):
    """Returns an interpolation function"""
    if spherical and gaussian:
        return slerp_gaussian
    elif spherical:
        return slerp
    elif gaussian:
        return lerp_gaussian
    else:
        return lerp


"""
#Example SLERP:
low = np.ones(200)
high = np.arange(200)
val = .5 #.5 to get midpoint
interpolated = slerp(val, low, high)
print(interpolated)
"""





"""
zpath = r"chairs/save-16K_zvectors.npy"
savename = 'interpolated_chairs_37_39'
zvectors = np.load(zpath)
#print(zvectors.shape)
#Determined manually from plotting
z1 = zvectors[37]
z2 = zvectors[39]
"""




"""
#AIRPLANES
zpath = r"airplane/save-airplane10400_zvectors.npy"
savename = 'interpolated_airplanes_61_37'
zvectors = np.load(zpath)
z1 = zvectors[61]
z2 = zvectors[37]
"""




#Final report: many Chairs
zpath = r"../chairs_480pts/save-16K_15batches_zvectors.npy"
savename = 'interpolated_chairs_403_59'
zvectors = np.load(zpath)
z1 = zvectors[403]
z2 = zvectors[59]


val_increment = 1./31. #use 31 sothattotal Npoints is same as batchsize 32 #0.05
current_val = np.arange(0.,1.+val_increment,val_increment)
#current_val = 0.
current_interpolation = None
#while current_val <= 1.0:
for i in current_val:
#  low = np.ones(200)
#  high = np.arange(200)
  interpolated = slerp(i, z1, z2).reshape((200, 1))

  if current_interpolation is None:
      current_interpolation = interpolated
  else:
      current_interpolation = np.concatenate((current_interpolation, interpolated), axis=1)
  #current_val += val_increment
    
    
current_interpolation = current_interpolation.T
print(z1[:3])
print(current_interpolation)
print(z2[:3])
print(current_interpolation.shape)
np.save(savename, current_interpolation)
