# [ICCV 2025] HOLa: Zero-Shot HOI Detection with Low-Rank Decomposed VLM Feature Adaptation

## Paper Links

[arXiv](https://arxiv.org/abs/2507.15542) 

[Project Page](https://chelsielei.github.io/HOLa_Proj/)




## Dataset 
Follow the process of [UPT](https://github.com/fredzzhang/upt).

The downloaded files should be placed as follows. Otherwise, please replace the default path to your custom locations.
```
|- HOLa
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |- vcoco
|   |   |- mscoco2014
|   |       |- train2014
|   |       |-val2014
:   :      
```

## Dependencies
1. Follow the environment setup in [UPT](https://github.com/fredzzhang/upt).

2. Follow the environment setup in [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main).

**Reminder**: 
If you have already installed the clip package in your Python environment (e.g., via pip install clip), please ensure that you use the local CLIP directory provided in our EZ-HOI repository instead. To do this, set the  `PYTHONPATH` to include the local CLIP path so that it takes precedence over the installed package.
```
export PYTHONPATH=$PYTHONPATH:"your_path/HOLa/CLIP"
```
So that you can use the local clip **without uninstall the clip of your python env**.

3. modify the installed [pocket](https://github.com/fredzzhang/pocket) library as mentioned [here](https://github.com/ChelsieLei/EZ-HOI/issues/2)

## Scripts
### Train / Test on HICO-DET:

Using vit-B image backbone:
```
bash scripts/hico_vitB.sh
```

Using vit-L image backbone:
```
bash scripts/hico_vitL.sh
```


### Train / Test on V-COCO:

Using vit-L image backbone:
```
bash scripts/vcoco.sh
```


## Model Zoo

| Dataset | Setting| Backbone  | mAP | Unseen | Seen |
| ---- |  ----  | ----  | ----  | ----  | ----  |
| HICO-DET | UV | ResNet-50+ViT-B  | 34.09|27.91|35.09|
| HICO-DET | RF| ResNet-50+ViT-B  | 34.19 |30.61|35.08|
| HICO-DET | NF| ResNet-50+ViT-B  | 32.36|35.25|31.64|
| HICO-DET | UO| ResNet-50+ViT-B  | 33.59|36.45|33.02|

| Dataset | Setting| Backbone  | mAP | Rare | Non-rare |
| ---- |  ----  | ----  | ----  | ----  | ----  |
| HICO-DET |default| ResNet-50+ViT-B  | 35.41|34.35|35.73|
| HICO-DET |default| ResNet-50+ViT-L  | 39.05|38.66|39.17|


You can download our pretrained model checkpoints using the following link from Google Drive:  
```
https://drive.google.com/drive/folders/1kH-yOi-YqdB35rSgKoRkmg_pGbyFEkUX?usp=sharing
```

You can also download our pretrained model checkpoints using the following Kuake link:  
```
Link: https://pan.quark.cn/s/c3f30b122ed2 
Extraction code: yawa
```

## Citation
If you find our paper and/or code helpful, please consider citing :
```
@inproceedings{
lei2025hola,
title={HOLa: Zero-Shot HOI Detection with Low-Rank Decomposed VLM Feature Adaptation},
author={Lei, Qinqian and Wang, Bo and Robby T., Tan},
booktitle={In Proceedings of the IEEE/CVF international conference on computer vision},
year={2025}
}
```

## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt) and [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main) for open-sourcing their code.






