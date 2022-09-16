#!/bin/bash
cd /ec2-user/repos/mmclassification
tensorboard --logdir work_dirs/"$1" --host=0.0.0.0 --port=6007 &
python tools/train.py configs/swin_transformer/swin-finetune.py --work-dir work_dirs/"$1" --resume-from work_dirs/"$1"/latest.pth
