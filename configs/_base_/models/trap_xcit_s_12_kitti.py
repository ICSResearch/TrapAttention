_base_ = [
    '../models/trap.py', '../datasets/kitti.py',
    '../default_runtime.py'
]

# norm_cfg = dict(type='BN', requires_grad=True)
norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    decode_head=dict(
        min_depth=1e-3,
        max_depth=80,
        norm_cfg=norm_cfg,
    ),
    )

find_unused_parameters=True
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