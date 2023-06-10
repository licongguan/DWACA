#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='eval'
cfg='gtav/deeplabv3plus_r101_RA_wt_5%.yaml'
experiment='v3plus_gtav_ra_wt_5.0_precent'

mkdir -p results/$experiment/val


python test.py \
    -cfg configs/$cfg \
    resume results/$experiment/model_iter040000.pth \
    OUTPUT_DIR results/$experiment/val 2>&1 | tee results/$experiment/val/val_$now.txt
