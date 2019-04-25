# Stylegan on PyTorch

My implementation of StyleGAN on PyTorch, up to 256x256 generator.

- [Original Paper](https://arxiv.org/abs/1812.04948)
- [Official implementation(TensorFlow)](https://github.com/NVlabs/stylegan)

## Run

### Docker image
There's a Dockerfile including all requirements.
NGC Account is required for base image.

https://www.nvidia.com/en-us/gpu-cloud/

### Settings

By default, the networks can generate up to 256x256 images.

1. Place images under `../images/{label}`
1. Edit settings.json
1. Run `python train_gay.py`

Directory structure must be like below:

```
├─ images
|  ├─ ffhq
|  | ├─ image1.png
|  | └─ ...
|  ├─ your custom label1
|  | ├─ image1.png
|  | └─ ...
|  ├─ your custom label2
|  | ├─ image1.png
|  | └─ ...
|  └─ ...
└─ stylegan(this repository)

```

## Result

![face](https://user-images.githubusercontent.com/12446914/56738130-f12cfc00-67a6-11e9-93ea-95abd08d5418.png)

## Training time

It took about 3 days to train 256x256 image generator on single RTX2080 machine.
