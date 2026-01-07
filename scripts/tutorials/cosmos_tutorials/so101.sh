#!/bin/bash
set -e

# =============================================================================
# Configuration
# =============================================================================

# Directory containing Cosmos Predict 2.5 inference outputs.
# This directory should include generated videos produced by the Cosmos model.
LEROBOT_INPUT="<path_to_cosmos-predict2.5/outputs/>"

# Working directory used to store all intermediate and final IDM outputs.
# It is recommended to place this on a fast local disk.
WORK_DIR="<path_to_IDM_workdir>"

# Robot embodiment type (used by IDM and LeRobot).
ROBOT_TYPE="so101"

# Key name used to store videos in the LeRobot dataset structure.
# This should match the observation key expected by downstream IDM scripts.
VIDEO_KEY="observation.images.front"

# =============================================================================
# Intermediate directories (auto-generated)
# =============================================================================

# Step 1 output: task-named directories converted from Cosmos outputs
STEP1_DIR="${WORK_DIR}/step1"

# Step 2 output: split videos and text instructions
STEP2_DIR="${WORK_DIR}/step2"

# Step 3 output: preprocessed videos (e.g., resized, normalized)
STEP3_DIR="${WORK_DIR}/step3"

# Step 4 output: final LeRobot-format dataset
STEP4_DIR="${WORK_DIR}/${ROBOT_TYPE}.data"

# =============================================================================
# Step 1: Convert Cosmos outputs to task-based directory structure
# =============================================================================
# - Reads Cosmos Predict outputs
# - Groups videos by task name
# - Prepares data for downstream preprocessing
python3 IDM_dump/scripts/preprocess_leisaac/cosmos2.5_to_step2_format.py \
    --cosmos_dir "${LEROBOT_INPUT}" \
    --output_dir "${STEP1_DIR}"

# =============================================================================
# Step 2: Split videos and instructions
# =============================================================================
# - Separates raw videos and text instructions into:
#     - videos/
#     - labels/
# - The --recursive flag allows processing nested task directories
python3 IDM_dump/scripts/preprocess_leisaac/split_video_instruction.py \
    --source_dir "${STEP1_DIR}" \
    --output_dir "${STEP2_DIR}" \
    --recursive

# =============================================================================
# Step 3: Preprocess videos
# =============================================================================
# - Resizes videos to the resolution expected by IDM
# - Converts video format if necessary
# - Preserves directory structure across tasks
python3 IDM_dump/scripts/preprocess_leisaac/preprocess_video.py \
    --src_dir "${STEP2_DIR}" \
    --dst_dir "${STEP3_DIR}" \
    --dataset "${ROBOT_TYPE}" \
    --original_width 640 \
    --original_height 480 \
    --recursive

# =============================================================================
# IMPORTANT USAGE NOTE
# =============================================================================
# It is STRONGLY RECOMMENDED to:
#
#   1. Run Step 1–3 first
#   2. Inspect the contents of ${STEP3_DIR}
#   3. Identify the generated task directory name(s)
#   4. Manually copy the desired task directory name into Step 4
#
# This avoids hard-coding task names before they are known and
# allows flexible reuse of this script across different tasks.
#
# Example:
#   ls ${STEP3_DIR}
#   → Lift_the_red_cube_up
#
# Then use:
#   --input_dir "${STEP3_DIR}/Lift_the_red_cube_up"
#
# =============================================================================

# =============================================================================
# Step 4: Convert preprocessed data to LeRobot dataset reminder
# =============================================================================
# --input_dir:
#   Path to a task-specific directory under STEP3_DIR.
#   The directory name MUST match the task name generated in Step 3.
#
# --output_dir:
#   Target directory for the LeRobot-format dataset.
#
# --fps:
#   Target frames per second for the output dataset.
#
# --embodiment:
#   Robot embodiment identifier used by LeRobot/IDM.
#
# --video_key:
#   Observation key used to store video data.
python3 IDM_dump/scripts/preprocess_leisaac/raw_to_lerobot.py \
    --input_dir "${STEP3_DIR}/Lift_the_red_cube_up" \
    --output_dir "${STEP4_DIR}" \
    --fps 16 \
    --embodiment "${ROBOT_TYPE}" \
    --video_key "${VIDEO_KEY}" \
    --embodiment "so101"

# =============================================================================
# Step 5: Dump IDM actions from the LeRobot dataset
# =============================================================================
# - Loads a pretrained IDM checkpoint
# - Runs IDM inference on the LeRobot dataset
# - Exports predicted action trajectories
#
# --checkpoint:
#   Path to a trained IDM checkpoint.
#
# --dataset:
#   Path to the LeRobot dataset generated in Step 4.
#
# --output_dir:
#   Output directory where IDM predictions will be stored.
#
# --num_gpus:
#   Number of GPUs used for IDM inference.
#
# --video_indices:
#   Indices of videos to process (e.g., "0 16" processes videos 0–16).
python3 IDM_dump/scripts/preprocess_leisaac/dump_idm_actions.py \
    --checkpoint "<path_to_the_trained_IDM_checkpoint>" \
    --dataset "${STEP4_DIR}" \
    --output_dir "${STEP4_DIR}_idm_cosmos" \
    --num_gpus 1 \
    --video_indices "0 16"
