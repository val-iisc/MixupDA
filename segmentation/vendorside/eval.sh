#!/bin/bash

# BASE='/workspace/machine54'
# BASE='/home/akshay/remote/54'
# BASE='/home/cds/akshay/machine54'
# BASE='/data/akshay/machine54'
BASE='/sda'

CUDA_VISIBLE_DEVICES=0 python3 eval.py --dataset="cityscapes" --load_model="checkpoints/dl_allg_gs_lovasz.pth" \
--base=$BASE
