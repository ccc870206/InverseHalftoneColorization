# Inverse Halftone Colorization: Making Halftone Prints Color Photos (ICIP 2021 accepted)
PyTorch implementaton of the following paper. In this paper, we propose a framework to recover colorful images from black and white halftone prints. You can visit our project website [here](https://ccc870206.github.io/InverseHalftoneColorization/).

![img](/figure/teaser.png)

## Paper
[Inverse Halftone Colorization: Making Halftone Prints Color Photos](https://ieeexplore.ieee.org/document/9506307)  
Yu-Ting Yen, Chia-Chi Cheng, [Wei-Chen Chiu](https://walonchiu.github.io/)  
IEEE International Conference on Image Processing (ICIP), 2021.

Please cite our paper if you find it useful for your research.  
```
@InProceedings{yen2021inverse,
    title={Inverse Halftone Colorization: Making Halftone Prints Color Photos},
    author={Yen, Yu-Ting and Cheng, Chia-Chi and Chiu, Wei-Chen},
    booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
    pages={1734--1738},
    year={2021},
    organization={IEEE}
    }
```

## Installation
* This code was developed with Python 3.8.5 & Pytorch 1.4.0 & CUDA 11.3.
* Other requirements: numpy, Pillow
* Clone this repo
```
git clone https://github.com/ccc870206/InverseHalftoneColorization.git
cd InverseHalftoneColorization
```

## Testing
Download our pretrained models from [here](https://drive.google.com/drive/folders/1LJ5wQmoz0iovj6w0BoHeCXrT31HbsgT7?usp=share_link) and put them under `weights/`.  
Run the sample data provided in this repo:
```
python test.py
```
Run your own data:
```
python test.py --input_dir YOUR_INPUT_IMG_PATH
               --ref_dir YOUR_REFERENCE_IMG_PATH
               --target_img_path YOUR_TARGET_IMG_PATH
```

## Training (TODO)

## Acknowledgments
Our code is based on [BicycleGAN](https://github.com/junyanz/BicycleGAN) and we re-implement [Deep Inverse Halftoning via Progressively Residual Learning](https://github.com/MenghanXia/InverseHalftoning) in Pytorch for inverse halftone network.
