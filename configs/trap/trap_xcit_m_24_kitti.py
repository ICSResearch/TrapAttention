_base_ = [
    '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py'
]

norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='XCiT',
        pretrained=r'',
        patch_size=8,
        embed_dim=512,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        eta=1e-5,
        out_indices=range(24),
    ),
    neck=dict(
        type='BlockSelectionNeck',
        in_channels=[512] * 5,
        out_channels=[128, 192, 384, 768, 1536],
        start=[0, 4, 8, 12, 16],
        end=[8, 12, 16, 20, 24],
        scales=[4, 2, 1, .5, .25]),
    decode_head=dict(
        type='TrappedHead',
        in_channels=[128, 192, 384, 768, 1536],
        post_process_channels=[128, 192, 384, 768, 1536],
        channels=64,  # last one
        final_norm=False,
        scale_up=True,
        align_corners=False,  # for upsample
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=10)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )

find_unused_parameters = True
SyncBN = True

# batch size
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
)

# schedules
# optimizer
max_lr = 0.00001
optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=3200,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

optimizer_config = dict()
# runtime settings
# runner = dict(type='IterBasedRunnerAmp', max_iters=320000)
runner = dict(type='IterBasedRunner', max_iters=320000)

checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(by_epoch=False,
                  start=0,
                  interval=1600,
                  pre_eval=True,
                  rule='less',
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"),
                  less_keys=("abs_rel", "rmse"))

# iter runtime
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])