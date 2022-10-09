#!/bin/bash
cd /ec2-user/repos/mmclassification
tensorboard --logdir work_dirs/"$2" --host=0.0.0.0 --port=6007 &
python tools/train.py configs/internship/vision_transformer/vit-finetune-"$1".py --work-dir work_dirs/"$2"
