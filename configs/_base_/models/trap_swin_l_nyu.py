_base_ = [
    '../datasets/nyu.py',
    '../default_runtime.py'
]

backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='Swin',
        pretrained=r'',
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=range(24),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.5,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        pretrain_style='official',
    ),
    neck=dict(
        type='BlockSelectionNeck',
        in_channels=[192, 384, 768, 768, 1536],
        out_channels=[192, 288, 576, 1152, 2304],
        start=[0, 2, 4, 10, 22],
        end=[2, 4, 16, 22, 24],
        scales=[2, 2, 2, 1., 1.]),
    decode_head=dict(
        type='TrappedHead',
        in_channels=[192, 288, 576, 1152, 2304],
        post_process_channels=[192, 288, 576, 1152, 2304],
        channels=96,  # last one
        final_norm=False,
        scale_up=True,
        align_corners=False,  # for upsample
        min_depth=1e-3,
        max_depth=10,
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
max_lr = 0.0001


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