#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main_edge_mixup.py --batch_size 8 --start_iter 0 --end_iter 100000 --load_saved=True \
--load_prev_model dl_allg_gy_lovasz.pth --model deeplab --dataset gta5 synthia --runs edge_mixup_full_model/dl_allg_gy \
--save_current_model edge_mixup_full_model/dl_allg_gy.pth --save_every 100 --lr 2.5e-4 --mixup_lambda 0.005
