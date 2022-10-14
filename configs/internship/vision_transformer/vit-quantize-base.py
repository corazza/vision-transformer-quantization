_base_ = [
    '../_base_/models/vision_transformer/finetune_visage_vit.py',
    '../_base_/datasets/quantize_tiny_imagenet.py',
    '../_base_/schedules/finetune_visage.py',
    '../_base_/default_runtime_visage_finetune.py'
]
