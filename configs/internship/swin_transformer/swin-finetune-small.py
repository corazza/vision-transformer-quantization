_base_ = [
    '../_base_/models/swin_transformer/finetune_visage_small.py',
    '../_base_/datasets/finetune_imagenet.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin_finetune_visage.py',
    '../_base_/default_runtime_swin_visage_finetune.py'
]
