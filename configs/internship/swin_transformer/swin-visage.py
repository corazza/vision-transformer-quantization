_base_ = [
    '../_base_/models/swin_transformer/tiny_visage.py',
    '../_base_/datasets/tiny_imagenet.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin_visage.py',
    '../_base_/default_runtime_swin_tiny_visage.py'
]
