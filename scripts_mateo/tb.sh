#!/bin/bash
tensorboard --logdir "$1" --host=0.0.0.0 --port=6007
