_base_ = [
    '../_base_/models/swin_transformer/finetune_visage_base.py',
    '../_base_/datasets/finetune_imagenet.py',
    '../_base_/schedules/finetune_visage.py',
    '../_base_/default_runtime_visage_finetune.py'
]
