# EA-BEV


## News
+ **2023.4.6**  create README.md

 ![EA-BEV](./figure2.pdf)
 
## Main Result
### nuScenes detection test
| Method                                                                   | mAP        | NDS        |
| ------------------------------------------------------------------------- | ---------- | ---------- |
| **BEVFusion(Peking University)**    |  71.3       | 73.3       |
| [** +EA-BEV **](configs/bevfusion/bevf_tf_20e_nusc_cam_lr.py)     | 71.8      | 73.6    |

### nuScenes detection validation
| Method                                                                    | mAP        | NDS        |  Latency(ms) |
| ------------------------------------------------------------------------- | ---------- | ---------- |--------------|
| **BEVDepth**    |  35.1       | 47.5       | 110.3 |
| ** +EA-BEV **   | **40.4**       | **48.2**   | 114.8 |
| **BEVFusion(MIT)**    |  68.5       | 71.4       | 119.2 |
| ** +EA-BEV **   | **69.4**       | **71.8**   | 123.6 |
| **BEVFusion(Peking University)**    |  69.6       | 72.1       | 190.3 |
| [** +EA-BEV **](configs/bevfusion/bevf_tf_20e_nusc_cam_lr.py)     | **70.3**      | **72.6**    | 194.9|

### nuScenes BEV map segmentation validation

| Method    | Drivable | Ped.Cross. | Walkway | Stop Line  | Carpark | Divider | Mean |
| ---------- | ---------- | ---------- |--------------| ---------- | ---------- | ---------- |--------------|
| **BEVFusion(MIT)**    |  85.5       | 60.5       | 67.6 | 52.0 | 57.0 | 53.7 | 62.7 |
| ** +EA-BEV **   | **85.8**  | **61.1**   | **68.0** | **52.3** | 56.8 | **54.5** | **63.1** | 

### Detect visualization results
![ ](./figure2.pdf)

## Use EA-BEV
### install and date preparation
For environment installation method, please refer to [BEVFusion](https://https://github.com/ADLab-AutoDrive/BEVFusion).

### beachmark Evaluation and Training

```shell
# training example for EA-BEV
# first train camera stream
./tools/dist_train.sh configs/EA-BEV/cam_stream/eabev_tf_4x8_20e_nusc_cam_lr.py 8
# then train LiDAR stream (using fade strategy)
./tools/dist_train.sh configs/EA-BEV/lidar_stream/transfusion_nusc_voxel_L.py 8
# then train BEVFusion
./tools/dist_train.sh configs/EA-BEV/eabev_tf_4x8_10e_nusc_aug.py 8

### evaluation example for bevfusion-pointpillar
./tools/dist_test.sh configs/EA-BEV/eabev_ft_4x8_10e_nusc_aug.py ./work_dirs/ea-bev_tf.pth 8 --eval bbox
```

## Acknowlegement
We sincerely thank the authors of [BEVFusion(Peking University)](https://https://github.com/ADLab-AutoDrive/BEVFusion),[BEVFusion(MIT)](https://github.com/mit-han-lab/bevfusion),[mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [TransFusion](https://github.com/XuyangBai/TransFusion), [LSS](https://github.com/nv-tlabs/lift-splat-shoot), [CenterPoint](https://github.com/tianweiy/CenterPoint) for open sourcing their methods.
