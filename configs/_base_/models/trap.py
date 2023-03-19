# model settings
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='XCiT',
        pretrained=r'',
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        eta=1.0,
        drop_path_rate=0.05,
        out_indices=range(12)
        ),
    neck=dict(
        type='BlockSelectionNeck',
        in_channels=[384] * 5,
        out_channels=[64, 96, 192, 384, 768],
        start=[2, 4, 6, 8, 10],
        end=[4, 6, 8, 10, 12],
        scales=[4, 2, 1, .5, .25]),
    decode_head=dict(
        type='TrappedHead',
        in_channels=[64, 96, 192, 384, 768],
        post_process_channels=[64, 96, 192, 384, 768],
        # up_sample_channels=[128, 256, 512, 1024, 2048],
        channels=32, # last one
        final_norm=False,
        scale_up=True,
        align_corners=False, # for upsample
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=10)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
