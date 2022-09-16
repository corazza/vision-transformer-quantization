_base_ = [
    '../_base_/models/swin_transformer/finetune_visage.py',
    '../_base_/datasets/finetune_tiny_imagenet.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin_finetune_visage.py',
    '../_base_/default_runtime_swin_tiny_visage_finetune.py'
]
