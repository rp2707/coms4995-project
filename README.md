## How to set up the cloud instance
Create an instance with the following specs:
1.  6 vCPUs, 22.5 GB memory
2.  1 x NVIDIA Tesla K80 GPU
3.  500 GB Disk

## Environment Specification
* Python 2.7.12
* CUDA 8.0
* [cudaDNN 6.0](https://developer.nvidia.com/rdp/cudnn-download)
* TensorFlow 1.4

## How to obtain the dataset on the instance
```shell
wget http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip 
```

## How to monitor GPU usage
```shell
nvidia-smi
```
## References
* [Using a GPU & TensorFlow on Google Cloud Platform](https://medium.com/google-cloud/using-a-gpu-tensorflow-on-google-cloud-platform-1a2458f42b0)
* [TensorFlow Common Installation Problems](https://www.tensorflow.org/install/install_linux#common_installation_problems)
* [ImportError: libcudnn when running a TensorFlow program](https://stackoverflow.com/questions/41991101/importerror-libcudnn-when-running-a-tensorflow-program)





## How to do latent space interpolation:
After training a model to checkpoint K:
Output some 3D models and their corresponding z vectors by doing (with iter 16400 as example):

python 3dgan_mit_biasfree.py --train False --ckpath models/biasfree_tfbn.ckpt-16400

This will make an npy array of the 3D models and another npy file of the z vectors that generated those models.

You will have to choose two 3D models to use. Then get their corresponding z vectors from the _zvectors.npy and use them as the 2 points in the interpolation.py code by editing the part:

zpath = r"chairs/save-16K_zvectors.npy"
savename = 'interpolated_chairs_37_39'
zvectors = np.load(zpath)
#print(zvectors.shape)
#Determined manually from plotting
z1 = zvectors[37]
z2 = zvectors[39]

This will output an array with 32 z vectors (using 32 since it is the batchsize in the tensorflow code).
Now you will need to use those pre-made z vectors in the tensorflow code to generate 3D models from them, DO this:

python 3dgan_mit_biasfree.py --train False --ckpath models/biasfree_tfbn.ckpt-16400 --interpolatd_zs interpolated_chairs_37_39.npy

where the last argument is the path to the 32 z vectors for the interpolated trajectory.

Running this will output the 32 models in 3D.

Then run the script VisualizeGeneratorOutput.py on the output 3D models to visualize them and save as PNG's.

Finally, you can use the website gifmaker.me to make a GIF from the PNG's.