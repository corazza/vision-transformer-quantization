#!/bin/bash
docker run -d --gpus all --shm-size=8g --volume ~:/ec2-user -w /ec2-user pytorchchanged /bin/bash scripts/_start_experiments.sh
