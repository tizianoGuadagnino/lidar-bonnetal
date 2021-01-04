#!/bin/bash
cd train/tasks/mask_regression/
python3 train.py -data_cfg config/labels/semantic-kitti.yaml --dataset ~/data/ --arch_cfg config/arch/darknet53.yaml
