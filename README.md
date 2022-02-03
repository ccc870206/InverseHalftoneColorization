# Inverse Halftone Colorization: Making Halftone Prints Color Photos (ICIP 2021 accepted)
PyTorch implementaton of the following paper. In this paper, we propose a framework to recover colorful images from black and white halftone prints.  
![img](https://github.com/ccc870206/InverseHalftoneColorization/blob/master/figure/teaser.png)

## Paper
[Inverse Halftone Colorization: Making Halftone Prints Color Photos](https://ieeexplore.ieee.org/document/9506307)  
Yu-Ting Yen, Chia-Chi Cheng, [Wei-Chen Chiu](https://walonchiu.github.io/)  
IEEE International Conference on Image Processing (ICIP), 2021.

Please cite our paper if you find it useful for your research.  
```
@INPROCEEDINGS{9506307,
  author={Yen, Yu-Ting and Cheng, Chia-Chi and Chiu, Wei-Chen},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={Inverse Halftone Colorization: Making Halftone Prints Color Photos}, 
  year={2021},
  volume={},
  number={},
  pages={1734-1738},
  doi={10.1109/ICIP42928.2021.9506307}}
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
Download our pretrained models from [here](https://drive.google.com/drive/folders/17L-5K1tVc7xR1wFi4rCSqTo0s4v5tdeX?usp=sharing) and put them under `weights/`.  
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

## TODO: Training
