#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main_edge_mixup.py --batch_size 4 --start_iter 0 --end_iter 50000 --load_saved=True \
--load_prev_model dl_gta5_orig.pth --model deeplab --dataset gta5 synscapes --runs dl_allg_gs_005 \
--save_current_model dl_allg_gs_005.pth --save_every 1000 --lr 2.5e-4 --mixup_lambda 0.005
