_base_ = ['../../../_base_/datasets/pipelines/rand_aug.py']

# dataset settings
dataset_type = 'CustomDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

CROP_SIZE=56 # (56 : 64 :: 224 :: 256)
# CROP_SIZE=64 # (56 : 64 :: 224 :: 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=CROP_SIZE,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # TODO FIXME probably unnecessary to comment out-- optimized to noops?
    # dict(
    #     type='Resize',
    #     size=(64, -1),
    #     backend='pillow',
    #     interpolation='bicubic'),
    # dict(type='CenterCrop', crop_size=CROP_SIZE),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='data/tiny-imagenet-200/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/tiny-imagenet-200/val',
        class_prefix='data/tiny-imagenet-200/train',
        ann_file='data/tiny-imagenet-200/val/val.txt',
        # ann_file=None,
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/tiny-imagenet-200/val',
        class_prefix='data/tiny-imagenet-200/train',
        ann_file='data/tiny-imagenet-200/val/val.txt',
        # ann_file=None,
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='accuracy')
