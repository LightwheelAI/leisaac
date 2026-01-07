import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import concurrent.futures
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

import subprocess

import multiprocessing
from multiprocessing import Pool
import math


CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def get_video_metadata(video_path):
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of", "json", video_path,
    ]

    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]
        
        # Parse frame rate
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den

        return {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.is_depth_map": False,
            },
        }
    except Exception as e:
        print(f"Error getting video metadata: {e}")
        return None

def dump_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Dump list of dictionaries as JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            json_str = json.dumps(item)
            f.write(json_str + '\n')

def json_dump(data: Dict[str, Any], path: Path, indent: int = 4) -> None:
    """Dump dictionary as JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def process_video_chunk(args):
    """Process a chunk of videos in parallel."""
    video_files, labels_dir, output_dir, cosmos_predict2, data_type, videos_dir, video_key = args
    results = []
    
    for video_file in video_files:
        video_id = video_file.stem
        label_file = labels_dir / f"{video_id}.txt"

        # Read annotation
        with open(label_file, "r") as f:
            annotation = f.read().strip()
        
        # Get video frame count (if not in cosmos_predict2 mode)
        if cosmos_predict2:
            frame_count = 93  # Fixed frame count for cosmos_predict2
        else:
            try:
                cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                      "-show_entries", "stream=duration,r_frame_rate", "-of", "json", str(videos_dir / video_key / f"{video_file}.mp4")]
                output = subprocess.check_output(cmd).decode()
                probe_data = json.loads(output)
                
                if "streams" in probe_data and len(probe_data["streams"]) > 0:
                    stream = probe_data["streams"][0]
                    
                    # Check if duration is available
                    if "duration" in stream:
                        duration = float(stream["duration"])
                        fps_str = stream.get("r_frame_rate", "30/1")
                        try:
                            num, den = map(int, fps_str.split("/"))
                            fps = num / den
                            frame_count = int(duration * fps)
                        except Exception:
                            # Default to 30fps if fps parsing fails
                            frame_count = int(duration * 30)
                    else:
                        # If no duration, use a default frame count
                        print(f"Warning: Could not determine duration for {video_file}, using default frame count")
                        frame_count = 93
                else:
                    # No streams found
                    print(f"Warning: No valid video streams found in {video_file}, using default frame count")
                    frame_count = 93
                
            except Exception as e:
                print(f"Error getting frame count for {video_file}: {e}")
                # Default frame count if all methods fail
                frame_count = 93
        
        # Ensure frame count is at least 1
        frame_count = max(1, frame_count)
        
        results.append((video_id, annotation, frame_count))
    
    return results


def copy_videos_parallel(video_copy_tasks, max_workers=16):
    """Copy multiple videos in parallel using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for source, dest in video_copy_tasks:
            futures.append(executor.submit(shutil.copy2, source, dest))
        
        # Wait for all copy operations to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error copying file: {e}")

def process_folder(args):
    """Process a single input folder."""
    folder_path, output_base_dir, annotation_source, fps, max_videos, num_workers_per_folder, cosmos_predict2, data_type, embodiment, video_key = args

    
    
    # Create folder-specific output directory
    folder_name = folder_path.name

    output_dir = output_base_dir / f"{embodiment}.{folder_name}"
    
    # Process this folder
    result = convert_raw_to_lerobot(
        raw_dir=folder_path,
        output_dir=output_dir,
        annotation_source=annotation_source,
        fps=fps,
        max_videos=max_videos,
        num_workers=num_workers_per_folder,
        cosmos_predict2=cosmos_predict2,
        data_type=data_type,
        video_key=video_key,
    )
    
    return folder_name, result

def convert_raw_to_lerobot(
    raw_dir: Path,
    output_dir: Path,
    annotation_source: str = "human",
    fps: int = 8,
    max_videos: int | None = None,
    num_workers: int = None,
    cosmos_predict2: bool = False,
    data_type: str = "lapa",
    video_key: str = None,
):
    """Convert raw dataset to LeRobot format."""

    
    
    # Setup directories
    videos_dir = raw_dir / "videos"
    labels_dir = raw_dir / "labels"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata directory
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Check for the new folder structure
    video_subfolders = [f for f in videos_dir.iterdir() if f.is_dir()]
    if video_subfolders:
        # New folder structure detected
        print(f"Detected new folder structure with subfolders: {[f.name for f in video_subfolders]}")
        
        # Get all unique video IDs across all subfolders
        all_video_ids = set()
        for folder in video_subfolders:
            video_files = folder.glob("*.mp4")
            all_video_ids.update(video_file.stem for video_file in video_files)
        
        # Convert to sorted list with numeric sorting
        # Try to sort as integers if all IDs are numeric, otherwise use string sort
        try:
            video_ids = sorted(list(all_video_ids), key=lambda x: int(x) if x.isdigit() else float('inf'))
        except ValueError:
            # Fallback to string sort if conversion fails
            video_ids = sorted(list(all_video_ids))
        
        # Create dummy video_files list with just the IDs
        video_files = [Path(video_id) for video_id in video_ids]
    else:
        raise ValueError("No video subfolders found in the input directory")
    
    if max_videos is not None:
        video_files = video_files[:max_videos]
        print(f"Processing only first {max_videos} videos for debugging")

    print(f"Processing {len(video_files)} videos in {videos_dir}")

    # Setup multiprocessing
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)  # Use half of available cores by default

    # Split videos into chunks for parallel processing
    chunk_size = math.ceil(len(video_files) / num_workers)
    video_chunks = [video_files[i:i + chunk_size] for i in range(0, len(video_files), chunk_size)]
    
    # Prepare arguments for parallel processing
    args_list = [
        (chunk, labels_dir, output_dir, cosmos_predict2, data_type, videos_dir, video_key)
        for chunk in video_chunks
    ]

    # Process videos in parallel
    total_frames = 0
    annotation_to_index = {}
    episodes_info = []
    
    print(f"Processing {len(video_files)} videos in {raw_dir} using {num_workers} workers")
    print(f"cosmos_predict2 mode: {cosmos_predict2} (fixed FPS=16, frames=93)" if cosmos_predict2 else "")
    with Pool(num_workers) as pool:
        all_results = list(tqdm(
            pool.imap(process_video_chunk, args_list),
            total=len(args_list),
            desc=f"Processing {raw_dir.name}"
        ))

    # Process results and create videos
    print(f"Processing {len(all_results)} chunks of videos...")
    
    # Flatten all results and sort by video_id to ensure episode_index matches video order
    all_video_results = []
    for chunk_results in all_results:
        all_video_results.extend(chunk_results)
    
    # Sort by video_id numerically to ensure episode 0 -> video 1, episode 1 -> video 2, etc.
    try:
        all_video_results.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
    except (ValueError, IndexError):
        # Fallback to string sort if conversion fails
        all_video_results.sort(key=lambda x: x[0])
    
    # We'll collect all video copy tasks here
    all_video_copy_tasks = []
    
    
    for video_id, annotation, frame_count in tqdm(
        all_video_results,
        desc="Processing videos in order"
    ):
        if annotation not in annotation_to_index:
            annotation_to_index[annotation] = len(annotation_to_index)
        
        episode_index = len(episodes_info)
        
        # If cosmos_predict2 mode is enabled, use fixed frame count
        actual_frame_count = 93 if cosmos_predict2 else frame_count
        actual_fps = fps  # Use the actual fps parameter
        
        # Determine state/action dimensions based on data_type
        if data_type == "piper" or "piper" in str(output_dir).lower():
            state_action_dim = 8  # Piper: 6 joints + 2 grippers
        elif data_type == "so101" or "so101" in str(output_dir).lower():
            state_action_dim = 6  # SO101: 5 joints + 1 gripper
        else:
            state_action_dim = 44  # Default
        
        # Create episode data
        episode_data = {
            "observation.state": [np.zeros(state_action_dim, dtype=np.float32)] * actual_frame_count,
            "action": [np.zeros(state_action_dim, dtype=np.float32)] * actual_frame_count,
            "timestamp": [i/actual_fps for i in range(actual_frame_count)],
            "episode_index": [episode_index] * actual_frame_count,
            "index": np.arange(total_frames, total_frames + actual_frame_count),
            "task_index": [annotation_to_index[annotation]] * actual_frame_count,
            f"annotation.{annotation_source}": [[annotation_to_index[annotation]]] * actual_frame_count
        }
        
        # Save episode data
        episode_chunk = episode_index // CHUNKS_SIZE
        data_path = DATA_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)
        save_path = output_dir / data_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(episode_data)
        df.to_parquet(save_path)

       
        view_list = []
        for folder in video_subfolders:
            source_video_path = folder / f"{video_id}.mp4"
            if source_video_path.exists():
                view_list.append((folder.name, source_video_path))
            
        for video_key, source_video_path in view_list:
            # Get the LeRobot format key
            
            # Destination path in LeRobot format
            video_save_path = output_dir / VIDEO_PATH.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index
            )
            video_save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add to copy tasks instead of copying immediately
            all_video_copy_tasks.append((source_video_path, video_save_path))

        # Update episodes info
        episodes_info.append({
            "episode_index": episode_index,
            "tasks": [annotation],
            "length": actual_frame_count,
            "video_id": video_id
        })
        
        total_frames += actual_frame_count

    # Now copy all videos in parallel
    print(f"Copying {len(all_video_copy_tasks)} videos in parallel...")
    copy_videos_parallel(all_video_copy_tasks, max_workers=min(32, num_workers))

    # Generate metadata files
    # 1. tasks.jsonl
    tasks_path = meta_dir / "tasks.jsonl"
    tasks = [{"task_index": idx, "task": task} for task, idx in annotation_to_index.items()]
    dump_jsonl(tasks, tasks_path)

    # 2. episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    dump_jsonl(episodes_info, episodes_path)

    # 3. info.json
    # Find a sample video for metadata from each view
    
    # Determine robot-specific configuration
    if data_type == "piper" or "piper" in str(output_dir).lower():
        robot_type = "Piper-Abs"
        state_action_shape = (8,)
        state_action_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", 
                              "finger_joint_left", "finger_joint_right"]
    elif data_type == "so101" or "so101" in str(output_dir).lower():
        robot_type = "SO101"
        state_action_shape = (6,)
        state_action_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", 
                              "wrist_roll", "gripper"]
    else:
        robot_type = data_type
        state_action_shape = (44,)
        state_action_names = [f"motor_{i}" for i in range(44)]
    
    info = {
        "robot_type": robot_type,
        "total_episodes": len(video_files),
        "total_frames": total_frames,
        "total_tasks": len(annotation_to_index),
        "total_videos": len(video_subfolders),
        "chunks_size": CHUNKS_SIZE,
        "total_chunks": (len(video_files) + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        "fps": fps,  # Use the actual fps parameter
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": state_action_shape,
                "names": state_action_names
            },
            "action": {
                "dtype": "float32",
                "shape": state_action_shape,
                "names": state_action_names
            },
            f"annotation.{annotation_source}": {
                "dtype": "int64",
                "shape": (1,)
            }
        }
    }
    

    for folder in video_subfolders:
        # Find the first video in this folder
        sample_videos = list(folder.glob("*.mp4"))
        if sample_videos:
            view_key = folder.name
            sample_path = sample_videos[0]
            info["features"][view_key] = get_video_metadata(sample_path)

    
    # Remove None values from features
    info["features"] = {k: v for k, v in info["features"].items() if v is not None}
    
    info_path = meta_dir / "info.json"
    json_dump(info, info_path, indent=4)

    return output_dir

def process_multiple_folders(
    input_base_dir: Path,
    output_base_dir: Path,
    annotation_source: str = "human",
    fps: int = 8,
    max_videos: int | None = None,
    num_workers: int = None,
    cosmos_predict2: bool = False,
    data_type: str = "lapa",
    embodiment: str = None
):
    """Process multiple input folders in parallel."""
    # Get all subdirectories that contain videos/ and labels/ folders
    input_folders = []
    for folder in sorted(input_base_dir.iterdir()):
        if folder.is_dir() and (folder / "videos").exists() and (folder / "labels").exists():
            input_folders.append(folder)

    print(f"Found {len(input_folders)} folders to process")
    
    # Determine number of workers per folder
    total_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
    workers_per_folder = max(1, total_workers)
    
    # Prepare arguments for parallel folder processing
    args_list = [
        (folder, output_base_dir, annotation_source, fps, max_videos, workers_per_folder, cosmos_predict2, data_type, embodiment)
        for folder in input_folders
    ]
    
    # Process folders in parallel using ThreadPoolExecutor
    # (ProcessPoolExecutor would create too many processes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_folder, args): args[0] for args in args_list}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing folders"):
            folder_path = futures[future]
            try:
                folder_name, result = future.result()
                print(f"Completed processing folder: {folder_name}")
            except Exception as e:
                print(f"Error processing folder {folder_path.name}: {e}")
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Convert raw dataset to LeRobot format")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing multiple folders with videos/ and labels/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=16, help="Video FPS")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process per folder (for debugging)")
    parser.add_argument("--num_workers", type=int, default=16, help="Total number of worker processes")
    parser.add_argument("--cosmos_predict2", action="store_true", help="Process videos for cosmos_predict2 video models (fixed FPS=8, frames=81)")
    parser.add_argument("--recursive", action="store_true", help="Process a single folder instead of multiple folders")
    parser.add_argument("--data_type", type=str, default="dream", choices=["lapa", "dream"])
    parser.add_argument("--embodiment", type=str, default=None, help="Embodiment")
    parser.add_argument("--video_key", type=str, default=None, help="Video key if cosmos_predict2 is false")

    args = parser.parse_args()

    if args.embodiment is None:
        if 'robocasa' in args.output_dir:
            args.embodiment = "robocasa_panda_omron"
        elif 'gr1' in args.output_dir:
            args.embodiment = "gr1_unified"
        elif 'franka' in args.output_dir:
            args.embodiment = "franka"
        elif 'so101' in args.output_dir:
            args.embodiment = "so101"
        elif 'so100' in args.output_dir:
            args.embodiment = "so100"
        elif 'piper' in args.output_dir:
            args.embodiment = "piper"
        elif 'piper double' in args.output_dir:
            args.embodiment = "piper double"
        else:
            raise ValueError(f"Unknown embodiment for {args.output_dir}")\

    if args.embodiment == "robocasa_panda_omron":
        args.annotation_source = "human.action.task_description"
    elif args.embodiment == "gr1_unified":
        args.annotation_source = "human.coarse_action"
    elif args.embodiment == "franka":
        args.annotation_source = "language.language_instruction"
    elif args.embodiment == "so100":
        args.annotation_source = "human.task_description"
    elif args.embodiment == "so101":
        args.annotation_source = "human.task_description"
    elif args.embodiment == "piper":
        args.annotation_source = "human.task_description"
    elif args.embodiment == "piper double":
        args.annotation_source = "human.task_description"
    
    if args.recursive:
        # Process a single folder (original behavior)
        process_multiple_folders(
            input_base_dir=Path(args.input_dir),
            output_base_dir=Path(args.output_dir),
            annotation_source=args.annotation_source,
            fps=args.fps,
            max_videos=args.max_videos,
            num_workers=args.num_workers,
            cosmos_predict2=args.cosmos_predict2,
            data_type=args.data_type,
            embodiment=args.embodiment
        )
    else:
        convert_raw_to_lerobot(
            raw_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            annotation_source=args.annotation_source,
            fps=args.fps,
            max_videos=args.max_videos,
            num_workers=args.num_workers,
            cosmos_predict2=args.cosmos_predict2,
            data_type=args.data_type,
            video_key=args.video_key
        )

    if args.embodiment == "gr1_unified": 
        source_dir = "IDM_dump/global_metadata/gr1"
    elif args.embodiment == "robocasa_panda_omron":
        source_dir = "IDM_dump/global_metadata/robocasa"
    elif args.embodiment == "franka":
        source_dir = "IDM_dump/global_metadata/franka"
    elif args.embodiment == "so100":
        source_dir = "IDM_dump/global_metadata/so100"
    elif args.embodiment == "so101":
        source_dir = "IDM_dump/global_metadata/so101"
    elif args.embodiment == "piper":
        source_dir = "IDM_dump/global_metadata/piper"
    elif args.embodiment == "piper double":
        source_dir = "IDM_dump/global_metadata/piper double"
    
    # copy modality.json
    shutil.copy(source_dir + "/modality.json", args.output_dir + "/meta/modality.json")

    # # copy stats.json
    shutil.copy(source_dir + "/stats.json", args.output_dir + "/meta/stats.json")
    

if __name__ == "__main__":
    main()