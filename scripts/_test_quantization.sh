#!/bin/bash
cd /ec2-user/repos/mmclassification
python tools/internship/test_quantization.py configs/internship/swin_transformer/swin-quantize-tiny.py work_dirs/finetune-relu/latest.pth --device cpu --model-out work_dirs/"$1"/quantized.pth --out work_dirs/"$1"/output.json --out-items none --metrics accuracy precision recall f1_score
