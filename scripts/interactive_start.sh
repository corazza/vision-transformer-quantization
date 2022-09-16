#!/bin/bash
docker run -it -p 0.0.0.0:6007:6007 --gpus all --shm-size=8g --volume ~:/ec2-user -w /ec2-user mmclassification
