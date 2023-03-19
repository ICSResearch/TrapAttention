# dataset settings
dataset_type = 'SUNRGBDDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SUNCrop', depth=False,  height=480, width=640),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        flip=False,
        flip_direction='horizontal',
        transforms=[
            # dict(type='RandomFlip', direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Resize', img_scale=(480, 640), keep_ratio=True),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 keys=['img'],
                 # keys=['img', 'depth_gt'],
                 # meta_keys=('cam_intrinsic', 'ori_shape', 'img_shape')
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor',
                            'flip', 'flip_direction', 'img_norm_cfg',
                            'cam_intrinsic')
                 ),
        ])
]
eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='SUNCrop', depth=False,  height=480, width=640),
    dict(type='RandomFlip', prob=0.0),  # set to zero
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect',
         keys=['img'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'flip', 'flip_direction', 'img_norm_cfg',
                    'cam_intrinsic')),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=1000,
        split='SUNRGBD_val_splits.txt',
        pipeline=test_pipeline,
        garg_crop=False,
        eigen_crop=True,
        min_depth=1e-3,
        max_depth=10),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        depth_scale=8000,
        split='SUNRGBD_val_splits.txt',
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=10))
