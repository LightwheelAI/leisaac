#!/bin/bash
TASK_NAME=${1:-LeIsaac-SO101-PickOrange-v0}
DATASET=${2:-./datasets/dataset_test.hdf5}
EPISODE_ID=${3:-1}

python scripts/environments/state_machine/replay.py \
    --dataset_file=${DATASET} \
    --task=${TASK_NAME} \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --select_episodes ${EPISODE_ID} \
    --replay_mode=action \
    --task_type=so101_state_machine