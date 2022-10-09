#!/bin/bash
cd /ec2-user/repos/mmclassification
python tools/internship/test_quantization.py configs/internship/vision_transformer/vit-finetune-base.py work_dirs/finetune-relu/latest.pth --device cpu --model-out work_dirs/"$1"/quantized.pth --out work_dirs/"$1"/output.json --out-items none --metrics accuracy precision recall f1_score
