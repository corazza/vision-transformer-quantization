#!/bin/bash
cd /ec2-user/repos/mmclassification
# tensorboard --logdir work_dirs/"$1" --host=0.0.0.0 --port=6007 &
python tools/internship/test_quantization.py configs/internship/swin_transformer/swin-quantize.py work_dirs/finetune-relu/latest.pth --alt-attn --device cpu --model-out work_dirs/"$1"/quantized.pth --out work_dirs/"$1"/output.json --out-items none --metrics accuracy precision recall f1_score
