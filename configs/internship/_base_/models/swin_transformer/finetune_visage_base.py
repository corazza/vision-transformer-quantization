model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='models/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth',
            prefix='backbone',
        ),
        type='SwinTransformerQ', arch='base', img_size=224, drop_path_rate=0.2, alt_attn=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=200,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=200, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=200, prob=0.5)
    ]))
    