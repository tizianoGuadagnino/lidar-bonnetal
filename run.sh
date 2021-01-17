#!/bin/bash
cd src
python3 train.py --dataset ~/data/ --arch_cfg config/arch/rangemasknet.yaml --data_cfg config/labels/semantic-kitti.yaml
