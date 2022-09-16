#!/bin/bash
cd /ec2-user/repos/mmclassification
# tensorboard --logdir work_dirs/"$1" --host=0.0.0.0 --port=6007 &
python tools/internship/test_quantization.py configs/internship/swin_transformer/swin-finetune.py work_dirs/tiny-finetune/latest.pth --out work_dirs/test_quantization/asdf.txt --out-items all
