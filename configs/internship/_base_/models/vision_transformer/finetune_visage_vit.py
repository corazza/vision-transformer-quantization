model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='models/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth',
            prefix='backbone',
        ),
        type='VisionTransformerQ', arch='b', img_size=224, patch_size=16, drop_rate=0.1),
    neck=None,
    head=dict(
        type='VisionTransformerClsHeadQ',
        num_classes=200,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=200, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=200, prob=0.5)
    ]))
    