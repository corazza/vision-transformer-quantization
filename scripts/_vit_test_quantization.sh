#!/bin/bash
cd /ec2-user/repos/mmclassification
python tools/internship/test_quantization.py configs/internship/vision_transformer/vit-quantize-"$1".py work_dirs/"$2"/latest.pth --device cpu --model-out work_dirs/"$2"/quantized.pth --out work_dirs/"$2"/output.json --out-items none --metrics accuracy precision recall f1_score
