# G-NAS: Generalizable Neural Architecture Search for Single Domain Generalization Object Detection [[AAAI24 Paper]](http://arxiv.org/abs/2402.04672)

![Algorithm framework](resources/algorithm_framework.png)

### Installation

Our code is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and please follow the [tutorial](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation.

Or you can just install this repository using the following commands:

```
conda create --name gnas python=3.8 -y
conda activate gnas
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/wufan-cse/gnas.git
cd gnas
pip install -v -e .
```

Notably, the pytorch and torchvision installation in the third line better follow the [official instructions](https://pytorch.org/get-started/locally).

### Datasets

Download the **Daytime-Sunny**, **Daytime-Foggy**, **Dusk-Rainy**, **Night-Sunny** and **Night-Rainy** datasets from this [link](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B).

国内用户可以通过[OpenDataLab](https://opendatalab.com/wufan/S-DGOD)下载。

Unzip and format the datasets as follows:

```
dataset_root_path/
    /daytime_clear
        /VOC2007
            /Annotations
            /ImageSets
                /Main
            /JPEGImages
    /daytime_foggy
    ...
```

### Training

We train our models on a V100 GPU platform.

##### 1. Search stage

Set the variable DATA_ROOT in [gnas_search_faster-rcnn_r101_fpn_1x_coco.py](https://github.com/wufan-cse/gnas/blob/main/configs/gnas/search/gnas_search_faster-rcnn_r101_fpn_1x_coco.py) to the dataset path, for example, DATA_ROOT='dataset_root_path/daytime_clear'.

```
# single gpu
python tools/train.py configs/gnas/search/gnas_search_faster-rcnn_r101_fpn_1x_coco.py

# multiple gpus
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/gnas/search/gnas_search_faster-rcnn_r101_fpn_1x_coco.py 2
```

##### 2. Augment stage

Similarly, set the variable DATA_ROOT in [gnas_augment_faster-rcnn_r101_fpn_1x_coco.py](https://github.com/wufan-cse/gnas/blob/main/configs/gnas/augment/gnas_augment_faster-rcnn_r101_fpn_1x_coco.py) to your dataset path.

```
# single gpu
python tools/train.py configs/gnas/augment/gnas_augment_faster-rcnn_r101_fpn_1x_coco.py

# multiple gpus
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/gnas/augment/gnas_augment_faster-rcnn_r101_fpn_1x_coco.py 2
```

### Evaluation

Please refer to the inference [instructions](https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html) for evaluating the saved model from the augment stage.

### New Results

Note: All results are running with FPN.

![Result table](resources/sdgod_results.png)

Full results in LaTeX format available [here](resources/sdgod_results.tex).

### Citation

```bibtex
@inproceedings{wu2024gnas,
  title = {G-NAS: Generalizable Neural Architecture Search for Single Domain Generalization Object Detection},
  author = {Wu, Fan and Gao, Jinling and Lanqing, HONG and Wang, Xinbing and Zhou, Chenghu and Ye, Nanyang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2024},
}
```
