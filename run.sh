#!/bin/bash
cd train/tasks/mask_regression/
python3 train.py --dataset ~/data/dataset/ --arch_cfg config/arch/darknet53.yaml
