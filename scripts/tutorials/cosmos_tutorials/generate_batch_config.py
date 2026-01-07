#!/usr/bin/env python3
"""
Generate batch inference configuration files (JSONL).

This script scans a directory of input videos and generates a JSONL file
for batch video2world inference. Each line corresponds to one video.
Optionally, task prompts can be loaded from episodes.jsonl.
"""

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Input video directory used to generate inference configs
VIDEO_DIR = "<path_to_lerobot_dataset>/videos/chunk-000/observation.images.front"

# Metadata directory (used when loading text prompts)
META_DIR = "<path_to_lerobot_dataset>/meta"

# Output JSONL file
OUTPUT_JSONL = "batch_inference_config.jsonl"

# Task name prefix
TASK_NAME = "liftcube"

# ---------------------------------------------------------------------
# Base configuration template shared by all videos
# ---------------------------------------------------------------------

BASE_CONFIG = {
    "inference_type": "video2world",
    "seed": 21,
    "guidance": 7,
    "resolution": "480,640",
    "enable_autoregressive": True,
    # Number of output frames (e.g. 110 ≈ 6s, 140 ≈ 8s, 210 ≈ 16s)
    "num_output_frames": 210,
    "chunk_size": 77,
    "chunk_overlap": 1,
    # Default prompt (will be overridden if episode-specific prompt is found)
    "prompt": "The robot arm is performing a task",
    "negative_prompt": (
        "The video captures a series of frames showing ugly scenes, static with no motion, "
        "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
        "pixelated images, poorly lit areas, underexposed and overexposed scenes, "
        "poor color balance, washed out colors, choppy sequences, jerky movements, "
        "low frame rate, artifacting, color banding, unnatural transitions, "
        "outdated special effects, fake elements, unconvincing visuals, "
        "poorly edited content, jump cuts, visual noise, and flickering. "
        "Overall, the video is of poor quality."
    ),
}

# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------

def main(use_prompt=True):
    video_dir = Path(VIDEO_DIR)
    meta_dir = Path(META_DIR)

    # Collect all mp4 video files
    video_files = sorted(video_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} video files")

    # Load prompts from episodes.jsonl if enabled
    episode_prompts = {}
    if use_prompt:
        episodes_file = meta_dir / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file, 'r') as f:
                for line in f:
                    episode_data = json.loads(line)
                    episode_index = episode_data.get("episode_index")
                    tasks = episode_data.get("tasks", [])
                    if episode_index is not None and tasks:
                        # Use the first task as the episode prompt
                        prompt = f"The robot arm is performing a task. {tasks[0]}"
                        episode_prompts[episode_index] = prompt
            print(f"✓ Loaded {len(episode_prompts)} prompts from episodes.jsonl")

    # Generate JSONL configuration file
    with open(OUTPUT_JSONL, 'w') as f:
        for video_file in video_files:
            # Video filename without extension (e.g. episode_000001)
            video_name = video_file.stem

            # Create per-video config based on the base template
            config = BASE_CONFIG.copy()
            config["name"] = f"{TASK_NAME}_{video_name}"
            config["input_path"] = str(video_file)

            # Assign episode-specific prompt if available
            if use_prompt:
                # Extract episode index from video filename
                episode_index = int(video_name.split('_')[-1])
                if episode_index in episode_prompts:
                    config["prompt"] = episode_prompts[episode_index]

            # Write one JSON object per line
            f.write(json.dumps(config) + '\n')

    # Summary information
    print(f"✓ Config file generated: {OUTPUT_JSONL}")
    print(f"  - Total videos: {len(video_files)}")
    print(f"  - Use prompts: {'Yes' if use_prompt else 'No'}")
    if use_prompt:
        print(
            f"  - Videos with matched prompts: "
            f"{sum(1 for vf in video_files if int(vf.stem.split('_')[-1]) in episode_prompts)}"
        )

# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # Text prompts are enabled only when '--use-prompt' is provided
    use_prompt = "--use-prompt" in sys.argv
    main(use_prompt=use_prompt)
