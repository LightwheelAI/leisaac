#!/usr/bin/env python3
"""
Convert LeRobot format dataset to convert_directory output format.

Input format (LeRobot):
    dataset_dir/
    ├── videos/chunk-000/observation.images.front/
    │   ├── episode_000000.mp4
    │   ├── episode_000001.mp4
    │   └── ...
    └── meta/
        ├── tasks.jsonl  (task definitions)
        └── episodes.jsonl  (episode -> task mapping)

Output format (after convert_directory):
    output_dir/
    ├── Lift_the_red_cube_up/
    │   ├── 0.mp4
    │   ├── 1.mp4
    │   └── ...
    └── Another_task/
        ├── 0.mp4
        └── ...

Usage:
    python lerobot_to_step2_format.py \
        --input_dir /path/to/lerobot/dataset \
        --output_dir /path/to/output \
        --video_key observation.images.front
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict


def load_tasks(tasks_file: Path) -> dict:
    """Load task definitions. Returns {task_index: task_description}"""
    tasks = {}
    with open(tasks_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            tasks[data['task_index']] = data['task']
    return tasks


def load_episodes(episodes_file: Path) -> list:
    """Load episode information. Returns list of {episode_index, tasks, length}"""
    episodes = []
    with open(episodes_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            episodes.append(data)
    return episodes


def sanitize_task_name(task: str) -> str:
    """
    Convert task description to valid directory name.
    Example: "Lift the red cube up." -> "Lift_the_red_cube_up"
    """
    task = task.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    task = task.replace(' ', '_')
    task = '_'.join(filter(None, task.split('_')))
    return task


def convert_lerobot_to_step2(
    input_dir: Path,
    output_dir: Path,
    video_key: str = "observation.images.front",
    chunk_name: str = "chunk-000"
):
    """Convert LeRobot format to convert_directory output format."""
    
    # Load tasks and episodes
    tasks = load_tasks(input_dir / "meta" / "tasks.jsonl")
    episodes = load_episodes(input_dir / "meta" / "episodes.jsonl")
    
    print(f"Loaded {len(tasks)} tasks")
    print(f"Loaded {len(episodes)} episodes")
    
    video_dir = input_dir / "videos" / chunk_name / video_key
    
    # Group episodes by task
    task_episodes = defaultdict(list)
    for episode in episodes:
        episode_idx = episode['episode_index']
        episode_tasks = episode.get('tasks', [])
        if episode_tasks:
            task = episode_tasks[0]
            task_episodes[task].append(episode_idx)
    
    print(f"\nStarting conversion...")
    
    total_copied = 0
    for task, episode_indices in task_episodes.items():
        # Create task directory
        task_dir_name = sanitize_task_name(task)
        task_dir = output_dir / task_dir_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTask: {task}")
        print(f"  Directory: {task_dir_name}")
        print(f"  Episodes: {len(episode_indices)}")
        
        # Copy video files
        for idx, episode_idx in enumerate(sorted(episode_indices)):
            src_video = video_dir / f"episode_{episode_idx:06d}.mp4"
            dst_video = task_dir / f"{idx}.mp4"
            shutil.copy2(src_video, dst_video)
            total_copied += 1
    
    print(f"\nConversion complete!")
    print(f"  Total videos copied: {total_copied}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot format to convert_directory output format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="LeRobot dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--video_key",
        type=str,
        default="observation.images.front",
        help="Video key name (default: observation.images.front)"
    )
    parser.add_argument(
        "--chunk_name",
        type=str,
        default="chunk-000",
        help="Chunk name (default: chunk-000)"
    )
    
    args = parser.parse_args()
    
    convert_lerobot_to_step2(
        Path(args.input_dir),
        Path(args.output_dir),
        args.video_key,
        args.chunk_name
    )


if __name__ == "__main__":
    main()
