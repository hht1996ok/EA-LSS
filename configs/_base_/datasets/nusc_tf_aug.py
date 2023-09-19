point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
evaluation = dict(interval=10)

dataset_type = 'NuScenesDataset'
data_root = '/dahuafs/groupdata/share/openset/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (1600, 896)
num_views = 6
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
radar_use_dims = [0, 1, 2, 6, 7, 8, 9]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=5,
        use_dim=radar_use_dims,
        max_num=1200
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # bev_aug
    dict(type='GlobalRotScaleTransBEV', resize_lim=(0.9, 1.1), rot_lim=(-0.785 , 0.785), trans_lim=0.5, is_train=True),
    dict(type='RandomFlip3DBEV'),

    dict(type='LoadMultiViewImageFromFiles'),
    #dict(type='ModalMask3D', mode='train'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'lidar_aug_matrix', 'radar'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=5,
        use_dim=radar_use_dims,
        max_num=1200
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        #pts_scale_ratio=[0.9, 0.95, 1.0, 1.05, 1.1],
        #flip=True,
        transforms=[
            dict(
                type='GlobalRotScaleTransBEV',
                resize_lim=(1.0, 1.0),
                rot_lim=(0.0, 0.0),
                trans_lim=0.0,
                is_train=False
            ),
            dict(type='RandomFlip3DBEV'),
            dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
            dict(type='MyNormalize', **img_norm_cfg),
            dict(type='MyPad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img', 'radar', 'lidar_aug_matrix'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        #type='CBGSDataset',
        #    dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + '/nuscenes_infos_train.pkl',
                load_interval=1,
                pipeline=train_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                box_type_3d='LiDAR'
        #    )
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
