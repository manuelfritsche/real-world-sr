# Real-World Super-Resolution
A PyTorch implementation of the [DSGAN](https://github.com/ManuelFritsche/real-world-sr/tree/master/dsgan) and [ESRGAN-FS](https://github.com/ManuelFritsche/real-world-sr/tree/master/esrgan-fs) models as described in the paper [Frequency Separation for Real-World Super-Resolution](https://arxiv.org/pdf/1911.07850.pdf). 
This work won the [AIM 2019](http://www.vision.ee.ethz.ch/aim19/) challenge on [Real-Wold Super-Resolution](https://arxiv.org/abs/1911.07783). For more information on the implementation visit the respective folders.

### Abstract
Most of the recent literature on image super-resolution (SR) assumes the availability of training data in the form of paired low resolution (LR) and high resolution (HR) images or the knowledge of the downgrading operator (usually bicubic downscaling). While the proposed methods perform well on standard benchmarks, they often fail to produce convincing results in real-world settings. This is because real-world images can be subject to corruptions such as sensor noise, which are severely altered by bicubic downscaling. Therefore, the models never see a real-world image during training, which limits their generalization capabilities. Moreover, it is cumbersome to collect paired LR and HR images in the same source domain.
To address this problem, we propose DSGAN to introduce natural image characteristics in bicubically downscaled images. It can be trained in an unsupervised fashion on HR images, thereby generating LR images with the same characteristics as the original images. We then use the generated data to train a SR model, which greatly improves its performance on real-world images. Furthermore, we propose to separate the low and high image frequencies and treat them differently during training. Since the low frequencies are preserved by downsampling operations, we only require adversarial training to modify the high frequencies. This idea is applied to our DSGAN model as well as the SR model. We demonstrate the effectiveness of our method in several experiments through quantitative and qualitative analysis. Our solution is the winner of the AIM Challenge on Real World SR at ICCV 2019.

### Pre-trained Models
| |[DSGAN](https://github.com/ManuelFritsche/real-world-sr/tree/master/dsgan)|[ESRGAN-FS](https://github.com/ManuelFritsche/real-world-sr/tree/master/esrgan-fs)|
|---|:---:|:---:|
|DF2K Gaussian|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/DSGAN/DF2K_gaussian.tar)|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DF2K_gaussian_SDSR.pth)/[TDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DF2K_gaussian_TDSR.pth)|
|DF2K JPEG|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/DSGAN/DF2K_jpeg.tar)|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DF2K_jpeg_SDSR.pth)/[TDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DF2K_jpeg_TDSR.pth)|
|DPED|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/DSGAN/DPED.tar)|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DPED_SDSR.pth)/[TDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/DPED_TDSR.pth)|
|AIM 2019|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/DSGAN/AIM2019.tar)|[SDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/AIM2019_SDSR.pth)/[TDSR](https://data.vision.ee.ethz.ch/timofter/FrequencySeparationRWSR/checkpoints/ESRGAN-FS/AIM2019_TDSR.pth)|

### BibTeX
    @inproceedings{fritsche2019frequency,
    author={Manuel Fritsche and Shuhang Gu and Radu Timofte},
    title ={Frequency Separation for Real-World Super-Resolution},
    booktitle={IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    year = {2019},
    }
