#!/bin/bash

ID=0
CUDA_VISIBLE_DEVICES=$ID python3 mst_eval.py --model="deeplab" --dataset="cityscapes" \
--load_model="./checkpoints/dl_gta5_orig.pth"
