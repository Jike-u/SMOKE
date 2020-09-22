[Original README](./README_ori.md)

## New features of this repo

- [ ] python code for KITTI evaluation
- [ ] multi-GPU inference
- [x] support training and evaluating on nuScenes dataset.
- [ ] tuned hyperparameters for nuScenes
- [ ] estimating objects' velocities and attributes
- [ ] radar data

## Requirements

All codes are tested under the following environment:

*   Ubuntu 16.04
*   Python 3.7
*   Pytorch 1.3.1
*   CUDA 10.1

## Dataset

This repo supports training and evaluating on official [KITTI 3D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and [nuScenes detection task](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any).
For KITTI, please first download the dataset and organize it as following structure:

```
kitti
│──training
│    ├──calib 
│    ├──label_2 
│    ├──image_2
│    └──ImageSets
└──testing
     ├──calib 
     ├──image_2
     └──ImageSets
```

For nuScenes, please follow the setup instruction in [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).

## Setup

1. We use `conda` to manage the environment:

```
conda create -n smoke python=3.7
conda activate smoke 
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch
pip install yacs scikit-image tqdm shapely nuscenes-devkit
```

2. Clone this repo:

```
git clone https://github.com/dashidhy/SMOKE.git
cd SMOKE
```

3. Build codes:

```
python setup.py build develop
```

4. Link to dataset directory:

```
mkdir datasets
ln -s /path/to/kitti datasets/kitti
ln -s /path/to/nuscenes datasets/nuscenes
```

## Getting started

First check the config files under `configs/`.  Following the origin setting, models are trained on 4 GPUs with 32 batch size, and tested on single GPU.

#### KITTI dataset:

Training:

```
python tools/plain_train_net_kitti.py --num-gpus 4 --config-file configs/smoke_gn_vector_kitti.yaml OUTPUT_DIR /your/output/dir
```

Evaluating:

```
python tools/plain_train_net_kitti.py --eval-only --config-file configs/smoke_gn_vector_kitti.yaml --ckpt /your/checkpoint OUTPUT_DIR /your/output/dir
```

#### nuScenes dataset

First, convert data of nuScenes to json file for better loading:

```
python tools/convert_nuscenes.py
```

Training:

```
python tools/plain_train_net_nusc.py --num-gpus 4 --config-file configs/smoke_gn_vector_nusc.yaml OUTPUT_DIR /your/output/dir
```

Evaluating, this script only writes json file as required in nuScenes detection task:

```
python tools/plain_train_net_nusc.py --eval-only --config-file configs/smoke_gn_vector_nusc.yaml --ckpt /your/checkpoint OUTPUT_DIR /your/output/dir
```

Compute metrics:

```
python tools/nusc_detection_eval.py --result_path /your/result/json/file --eval_set val --dataroot datasets/nuscenes --output_dir /your/output/dir
```