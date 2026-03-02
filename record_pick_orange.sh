#!/bin/bash
TASK_NAME=${1:-LeIsaac-SO101-PickOrange-v0}
DATASET=${2:-./datasets/dataset_test.hdf5}

python scripts/environments/state_machine/pick_orange.py \
    --dataset_file=${DATASET} \
    --task=${TASK_NAME} \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --num_demos=1