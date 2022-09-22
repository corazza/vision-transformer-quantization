#!/bin/bash
docker run -it --gpus all --shm-size=8g --volume ~:/ec2-user -w /ec2-user/repos/mmclassification test
