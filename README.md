<div align="center">

<h3> Dual Prototype Attention for Unsupervised Video Object Segmentation
 </h3> 
 <br/>
  <a href='https://arxiv.org/abs/2211.12036'><img src='https://img.shields.io/badge/ArXiv-2303.08314-red' /></a> 
  <br/>
  <br/>
<div>
    <a href='https://suhwan-cho.github.io' target='_blank'>Suhwan Cho* <sup> 1</sup></a>&emsp;
    <a href='https://hydragon.co.kr' target='_blank'>Minhyeok Lee* <sup> 1</sup> </a>&emsp;
    <a target='_blank'>Seunghoon Lee <sup> 1</sup></a>&emsp;
    <a href='https://dogyoonlee.github.io' target='_blank'>Dogyoon Lee <sup> 1</sup></a>&emsp;
    <a target='_blank'>Heeseung Choi <sup> 1,2</sup></a>&emsp;
    <a target='_blank'>Ig-Jae Kim <sup> 1,2</sup></a>&emsp;
    <a target='_blank'>Sangyoun Lee <sup>1,2</sup></a>&emsp;
</div>
<br>
<div>
                      <sup>1</sup> Yonsei University &nbsp;&nbsp;&nbsp;
                      <sup>2</sup> Korea Institute of Science and Technology (KIST) &nbsp;
</div>
<br>
<i><strong><a href='https://cvpr.thecvf.com' target='_blank'>CVPR 2024</a></strong></i>
<br>
<br>
</div>

## Abstract
Unsupervised video object segmentation (VOS) aims to detect and segment the most salient object in videos. The primary techniques used in unsupervised VOS are 1) the collaboration of appearance and motion information; and 2) temporal fusion between different frames. This paper proposes two novel prototype-based attention mechanisms, inter-modality attention (IMA) and inter-frame attention (IFA), to incorporate these techniques via dense propagation across different modalities and frames. IMA densely integrates context information from different modalities based on a mutual refinement. IFA injects global context of a video to the query frame, enabling a full utilization of useful properties from multiple frames. Experimental results on public benchmark datasets demonstrate that our proposed approach outperforms all existing methods by a substantial margin. The proposed two components are also thoroughly validated via ablative study.

 
## Datasets
Prepare all dataset.

- [DUTS](http://saliencydetection.net/duts)
- [DAVIS](https://davischallenge.org/davis2016/code.html)
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets)
- [YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects)

We use [RAFT](https://github.com/princeton-vl/RAFT) to generate optical flow maps.

You can also get pre-processed datasets from [TMO](https://github.com/suhwan-cho/TMO).

The complete dataset directory structure is as follows:
```
dataset dir/
├── DUTS_train/
│   ├── RGB/
│   │   ├── sun_ekmqudbbrseiyiht.jpg
│   │   ├── sun_ejwwsnjzahzakyjq.jpg
│   │   └── ...
│   └── GT/
│       ├── sun_ekmqudbbrseiyiht.png
│       ├── sun_ejwwsnjzahzakyjq.png
│       └── ...
├── DAVIS_train/
│   ├── RGB/
│   │   ├── bear_00000.jpg
│   │   ├── bear_00001.jpg
│   │   └── ...
│   ├── GT/
│   │   ├── bear_00000.png
│   │   ├── bear_00001.png
│   │   └── ...
│   └── FLOW/
│       ├── bear_00000.jpg
│       ├── bear_00001.jpg
│       └── ...
└── DAVIS_test/
    ├── blackswan/
    │   ├── RGB/
    │   │   ├── blackswan_00000.jpg
    │   │   ├── blackswan_00001.jpg
    │   │   └── ...
    │   ├── GT/
    │   │   ├── blackswan_00000.png
    │   │   ├── blackswan_00001.png
    │   │   └── ...
    │   └── FLOW/
    │       ├── blackswan_00000.jpg
    │       ├── blackswan_00001.jpg
    │       └── ...
    ├── bmx-trees
    └── ...
```

## Training Model
We use a two-stage learning strategy: pretraining and finetuning.

### Pretraining
1. Edit config.py. The data root path option and GPU index should be modified.
2. training
```
python pretrain.py
```

### Finetuning
1. Edit config.py. The best model path generated during the pretraining process is required.
2. training
```
python train_for_DAVIS.py
```

## Evaluation
See this [link](https://github.com/yongliu20/DAVIS-evaluation).

## Results 
Ours pre-calculated prediction masks can be downloaded [here](https://drive.google.com/file/d/1JSJ5mIu6Lq8l4aSEcrUAlVsBGdXeGgma/view?usp=drivesdk).
