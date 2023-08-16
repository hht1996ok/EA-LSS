# EA-LSS: Edge-aware Lift-splat-shot Framework for 3D BEV Object Detection
 ![EA-LSS](./photo/page2.png)
 
[paper](https://arxiv.org/abs/2303.17895)

## News
+ **2023.8.16**  create README.md
 
## Main Result
### nuScenes detection test
| Method                                                                   | mAP        | NDS        |
| ------------------------------------------------------------------------- | ---------- | ---------- |
| BEVFusion(Peking University)    |  71.3       | 73.3       |
| [**+EA-LSS**](configs/EABEV/eabev_tf_4x8_10e_nusc_aug.py)     | **72.2**     | **74.4**    |
| **+EA-LSS***     | **76.5**     | **77.6**    |
*reprsent the test time augment and model ensemble.

### nuScenes detection validation
| Method                                                                    | mAP        | NDS        |  Latency(ms) |
| ------------------------------------------------------------------------- | ---------- | ---------- |--------------|
| BEVDepth-R50    |  33.0       | 43.6       | 110.3 |
|  **+EA-LSS**   | **33.4**       | **44.1**   | 110.3 |
| Tig-bev    |  33.8       | 37.5       | 68.0 |
|  **+EA-LSS**   | **35.9**       | **40.7**   | 68.0 |
| BEVFusion(MIT)    |  68.5       | 71.4       | 119.2 |
|  **+EA-LSS**    | **69.4**       | **71.8**   | 123.6 |
| BEVFusion(Peking University)    |  69.6       | 72.1       | 190.3 |
|  [ **+EA-LSS**  ](configs/EABEV/eabev_tf_4x8_10e_nusc_aug.py)     | **71.2**      | **73.1**    | 194.9|


### Visualization results
#### nuScenes 3D object detection
![ ](./photo/page6.png)

## Use EA-LSS
### install and date preparation
For environment installation method, please refer to [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion).


## Acknowlegement
We sincerely thank the authors of [BEVFusion(Peking University)](https://github.com/ADLab-AutoDrive/BEVFusion), [BEVFusion(MIT)](https://github.com/mit-han-lab/bevfusion), [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [TransFusion](https://github.com/XuyangBai/TransFusion) for open sourcing their methods.
