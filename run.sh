#!/bin/bash
uname -a
#date
#env
date
CS_PATH='./dataset/LIP'
LR=7e-3
WD=5e-4
BS=4
GPU_IDS=0,1
RESTORE_FROM='./dataset/resnet101-imagenet.pth'
INPUT_SIZE='473,473'  
SNAPSHOT_DIR='./slip'
DATASET='train'
NUM_CLASSES=20
EPOCHS=150

#if [[ ! -e ${SNAPSHOT_DIR} ]]; then
#    mkdir -p  ${SNAPSHOT_DIR}
#fi

python train.py --data-dir ${CS_PATH} \
       --random-mirror\
       --random-scale\
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --epochs ${EPOCHS}


