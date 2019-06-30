#!/bin/bash

CS_PATH='./dataset/LIP'
BS=16
GPU_IDS='1'
INPUT_SIZE='473,473'
SNAPSHOT_FROM='./slip_glr0.007_2edge/lip_149.pth'
DATASET='val'
NUM_CLASSES=20

python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}
