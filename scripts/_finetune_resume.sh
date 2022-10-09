#!/bin/bash
cd /ec2-user/repos/mmclassification
tensorboard --logdir work_dirs/"$2" --host=0.0.0.0 --port=6007 &
python tools/train.py configs/internship/swin_transformer/swin-finetune-"$1".py --work-dir work_dirs/"$2" --resume-from work_dirs/"$2"/latest.pth
