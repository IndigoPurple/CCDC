# Cross-Camera Deep Colorization

<p align="left">
<img src="img/SIGMA.png">
</p>

tSinghua vIsual intelliGence and coMputational imAging lab ( [GitHub](https://github.com/THU-luvision) | [HomePage](http://www.luvision.net/) )


In this repository we provide code of the paper:
> **Cross-Camera Deep Colorization**

> Yaping Zhao, Haitian Zheng, Mengqi Ji, Ruqi Huang

> arxiv link: https://arxiv.org/abs/2209.01211

<p align="center">
<img src="img/CCDC.png">
</p>

# Usage
0. For pre-requisites, run:
```
conda env create -f environment.yml
conda activate ccdc
```
1. Pretrained model is currently available at [Google Drive](https://drive.google.com/drive/folders/12GwNwZZE6u359LFD6o7CGMz7P_-FQWul?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/16s3lTUcnK1oBmDSz5yJUrQ) (password: ql5l), download the `X4_30w.pth`, `X8_30w.pth` and put them in the `pretrained` folder.
 - `X4_30w.pth` is pretrained on the Vimeo dataset under the scale gap 4X. 
 - `X8_30w.pth` is pretrained on the Vimeo dataset under the scale gap 8X. 
 - If you want to train your own model, please prepare your own training set. 

2. For training,
   1. under the scale gap 4X, run:
    ```
    sh train_X4.sh
    ```
    or
    ```
    python train_ccdc.py  \
   --dataset demo   \
   --scale 4 \
   --display 100 \
   --batch_size 8  \
   --step_size 25000 \
   --gamma 0.5 \
   --loss CharbonnierLoss \
   --optim Adam \
   --lr 0.0001  \
   --checkpoints_dir ./exp_X4/ \
   --checkpoint_file ./pretrained/X4_30w.pth \
   --frame_num 2 \
   --with_GAN_loss 0 \
   --img_save_path result/colornetcp_exp4 \
   --net_type colornet1 \
   --pretrained 1 \
   --gpu_id 0 \
   --snapshot 5000
    ```
   2. under the scale gap 8X, run:
    ```
    sh train_X8.sh
    ```
    or
    ```
   python train_ccdc.py  \
   --dataset demo   \
   --scale 8 \
   --display 100 \
   --batch_size 8  \
   --step_size 25000 \
   --gamma 0.5 \
   --loss CharbonnierLoss \
   --optim Adam \
   --lr 0.0001  \
   --checkpoints_dir ./exp_X8/ \
   --checkpoint_file ./pretrained/X8_30w.pth \
   --frame_num 2 \
   --with_GAN_loss 0 \
   --img_save_path result/colornetcp_exp5 \
   --net_type colornet1 \
   --pretrained 1 \
   --gpu_id 0 \
   --snapshot 5000
    ```
3. For testing,
   1. under the scale gap 4X, run:
    ```
    sh test_X4.sh
    ```
    or
    ```
    python train_ccdc.py  \
   --mode test \
   --dataset demo   \
   --scale 4 \
   --display 100 \
   --batch_size 1  \
   --step_size 50000 \
   --gamma 0.5 \
   --loss CharbonnierLoss \
   --optim Adam \
   --lr 0.0001  \
   --checkpoints_dir ./exp_X4/ \
   --checkpoint_file ./pretrained/X4_30w.pth \
   --frame_num 2 \
   --with_GAN_loss 0 \
   --img_save_path result/ \
   --net_type colornet1 \
   --pretrained 0 \
   --gpu_id 0 \
   --snapshot 5000
    ```
   2. under the scale gap 8X, run:
    ```
    sh test_X8.sh
    ```
    or
    ```
   python train_ccdc.py  \
   --mode test \
   --dataset demo   \
   --scale 8 \
   --display 100 \
   --batch_size 1  \
   --step_size 50000 \
   --gamma 0.5 \
   --loss CharbonnierLoss \
   --optim Adam \
   --lr 0.0001  \
   --checkpoints_dir ./exp_X8/ \
   --checkpoint_file ./pretrained/X8_30w.pth \
   --frame_num 2 \
   --with_GAN_loss 0 \
   --img_save_path result/ \
   --net_type colornet1 \
   --pretrained 0 \
   --gpu_id 0 \
   --snapshot 5000
    ```


# Dataset
Dataset is stored in the folder `dataset/`, where subfolders `clean/`, `corrupted/`, `SISR/` contain ground truth HR images, corrupted LR images, upsampled LR images by interpolation (e.g., bicubic) or SISR methods.
Images in `SISR/` could be as same as in `corrupted/`, though preprocessing by advanced SISR methods (e.g., MDSR) brings a small performance boost.

`testlist.txt` and `trainlist.txt` could be modified for your experiment on other datasets. 

**This repo only provides a sample for demo purposes.**

# Citation
Cite our paper if you find it interesting!
```
@article{zhao2022cross,
  title={Cross-Camera Deep Colorization},
  author={Zhao, Yaping and Zheng, Haitian and Ji, Mengqi and Huang, Ruqi},
  journal={arXiv preprint arXiv:2209.01211},
  year={2022}
}
```