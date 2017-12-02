## How to set up the cloud instance
Create an instance with the following specs:
1. 6 vCPUs, 22.5 GB memory
2. 1 x NVIDIA Tesla K80 GPU
3. 500 GB Disk

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
