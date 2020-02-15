## ESRGAN-FS

This code is based on [BasicSR](https://github.com/xinntao/BasicSR).
For more information visit their repository.
Most of our changes are applied to `SRGAN_model.py`


## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard: 
  - PyTorch >= 1.1: `pip install tb-nightly future`
  - PyTorch == 1.0: `pip install tensorboardX`
  
## Get Started
Please see [wiki](https://github.com/xinntao/BasicSR/wiki/Training-and-Testing) for the basic usage, *i.e.,* training and testing.

## Pre-Trained Models
| |[ESRGAN-FS](https://github.com/ManuelFritsche/real-world-sr/tree/master/esrgan-fs)|
|---|:---:|
|DF2K Gaussian|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DF2K_gaussian_SDSR.pth)/[TDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DF2K_gaussian_TDSR.pth)|
|DF2K JPEG|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DF2K_jpeg_SDSR.pth)/[TDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DF2K_jpeg_TDSR.pth)|
|DPED|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DPED_SDSR.pth)/[TDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DPED_TDSR.pth)|
|AIM 2019|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/AIM2019_SDSR.pth)/[TDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/AIM2019_TDSR.pth)|
