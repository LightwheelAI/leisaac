"""
Convert a local LeRobot dataset folder (v2.1 episode-based layout) to a single HDF5 file.

Compared to the original version, this script can optionally:
  1) Sort frames inside each episode by a chosen column (default: "index" if present),
  2) Convert LeRobot normalized joint values (roughly degrees in [-100, 100]) into
     IsaacLab joint angles in RADIANS, using the same limits mapping as your replay-correct script.

Expected input layout (v2.1):
dataset_root/
├── data/chunk-000/episode_000000.parquet
├── data/chunk-000/episode_000001.parquet
├── videos/chunk-000/<camera_name>/episode_000000.mp4
└── meta/episodes.jsonl (optional) + other meta files

Output HDF5 layout (mirrors folder-like structure):
/
  attrs: source_dir, layout_version, created_at
  /meta/<filename>                 (text dataset, utf-8)
  /data/<chunk>/<episode>/...      (datasets from parquet columns)
  /videos/<chunk>/<camera>/<episode>.mp4  (uint8 bytes dataset, optional)

"""

import argparse
import datetime as dt
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

# Disable HDF5 file locking to avoid BlockingIOError on some filesystems
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

ISAACLAB_LIMITS_DEG = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10.0, 100.0),
]
LEROBOT_LIMITS_DEG = [
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (0.0, 100.0),
]


def denormalize_lerobot_to_isaaclab_radians(joint_values_lerobot: np.ndarray) -> np.ndarray:
    """
    LeRobot normalized degrees (roughly within LEROBOT_LIMITS_DEG) ->
    IsaacLab joint limits (degrees) -> radians.

    joint_values_lerobot:
      shape (T, D) where D >= 6.
      If D == 12, treated as bimanual (6+6).
      Otherwise, treated as single arm (first 6 processed, others converted deg->rad only).
    """
    joint_values_lerobot = np.asarray(joint_values_lerobot, dtype=np.float32)

    if joint_values_lerobot.ndim != 2:
        raise ValueError(f"Expected 2D array (T,D), got shape {joint_values_lerobot.shape}")

    dimension = joint_values_lerobot.shape[1]
    if dimension < 6:
        raise ValueError(f"Expected D>=6, got D={dimension}")

    result_deg = joint_values_lerobot.copy()

    if dimension == 12:
        # bimanual: apply same 6-dim mapping to left [0:6] and right [6:12]
        for arm_offset in (0, 6):
            for joint_index in range(6):
                isa_min, isa_max = ISAACLAB_LIMITS_DEG[joint_index]
                le_min, le_max = LEROBOT_LIMITS_DEG[joint_index]
                col = arm_offset + joint_index
                result_deg[:, col] = (result_deg[:, col] - le_min) / (le_max - le_min) * (isa_max - isa_min) + isa_min
    else:
        # single arm (or single arm + extra features): only map the first 6
        for joint_index in range(6):
            isa_min, isa_max = ISAACLAB_LIMITS_DEG[joint_index]
            le_min, le_max = LEROBOT_LIMITS_DEG[joint_index]
            result_deg[:, joint_index] = (result_deg[:, joint_index] - le_min) / (le_max - le_min) * (
                isa_max - isa_min
            ) + isa_min

    # degrees -> radians
    result_rad = result_deg * (np.pi / 180.0)
    return result_rad


def write_text_dataset(h5_group: h5py.Group, dataset_name: str, text: str) -> None:
    """Store UTF-8 text as variable-length string dataset."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    if dataset_name in h5_group:
        del h5_group[dataset_name]
    h5_group.create_dataset(dataset_name, data=text, dtype=string_dtype)


def read_file_as_uint8_bytes(file_path: Path) -> np.ndarray:
    data = file_path.read_bytes()
    return np.frombuffer(data, dtype=np.uint8)


def stack_object_column_to_numpy(column_values: Any) -> tuple[np.ndarray, str | None]:
    """
    Convert a parquet column to a numpy array.
    - If it's already a numpy array -> return directly
    - If it's a 1D array/list of per-row arrays/lists -> stack to (T, D)
    - If it's scalar-like -> return (T,) or scalar
    Returns: (array, note)
    """
    note = None

    if isinstance(column_values, np.ndarray):
        if (
            column_values.dtype == object
            and len(column_values) > 0
            and isinstance(column_values[0], (list, tuple, np.ndarray))
        ):
            try:
                stacked = np.stack([np.asarray(x) for x in column_values], axis=0)
                return stacked, note
            except Exception as exc:
                note = f"object-stack-failed: {exc}"
                return np.asarray(column_values), note
        return column_values, note

    if isinstance(column_values, (list, tuple)):
        if len(column_values) > 0 and isinstance(column_values[0], (list, tuple, np.ndarray)):
            try:
                stacked = np.stack([np.asarray(x) for x in column_values], axis=0)
                return stacked, note
            except Exception as exc:
                note = f"list-stack-failed: {exc}"
                return np.asarray(column_values, dtype=object), note
        return np.asarray(column_values), note

    return np.asarray(column_values), note


def write_numpy_dataset(
    h5_group: h5py.Group,
    dataset_name: str,
    array: np.ndarray,
    compression: str | None,
    compression_level: int | None,
) -> None:
    """Write/overwrite a dataset. If dtype=object, store JSON strings."""
    if dataset_name in h5_group:
        del h5_group[dataset_name]

    # Object dtype: store each row as json string
    if array.dtype == object:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        string_list = []
        for value in array.tolist():
            try:
                string_list.append(json.dumps(value, ensure_ascii=False))
            except Exception:
                string_list.append(str(value))
        h5_group.create_dataset(dataset_name, data=np.asarray(string_list, dtype=object), dtype=string_dtype)
        h5_group[dataset_name].attrs["stored_as"] = "json_string_list"
        return

    # Numpy unicode string dtype (e.g. <U21)
    if array.dtype.kind == "U":
        string_dtype = h5py.string_dtype(encoding="utf-8")
        h5_group.create_dataset(
            dataset_name,
            data=array.astype(object),
            dtype=string_dtype,
            compression=compression,
            compression_opts=compression_level,
            shuffle=True if compression else None,
        )
        return

    h5_group.create_dataset(
        dataset_name,
        data=array,
        compression=compression,
        compression_opts=compression_level,
        shuffle=True if compression else None,
    )


def load_episode_parquet(parquet_path: Path):
    import pandas as pd

    return pd.read_parquet(parquet_path)


def sort_episode_dataframe_inplace(dataframe, sort_column: str | None) -> str:
    """
    Sort dataframe by a chosen column if present.
    If sort_column is None, try common ordering keys in priority:
      index > frame_index > timestamp
    Returns the column used, or "" if not sorted.
    """
    candidate_columns = []
    if sort_column:
        candidate_columns.append(sort_column)
    candidate_columns += ["index", "frame_index", "timestamp"]

    for col in candidate_columns:
        if col in dataframe.columns:
            dataframe.sort_values(col, ascending=True, inplace=True)
            dataframe.reset_index(drop=True, inplace=True)
            return col

    return ""


def load_episode_data(dataset: LeRobotDataset, episode_idx: int) -> dict[str, Any]:
    """
    Manually load all frames for a specific episode from LeRobotDataset.
    Reconstructs the dictionary of arrays (T, D) for the episode.
    """
    start_frame = dataset.episode_data_index["from"][episode_idx].item()
    end_frame = dataset.episode_data_index["to"][episode_idx].item()

    # Pre-allocate lists for each key
    # We peek at the first frame to know the keys
    first_frame = dataset[start_frame]
    episode_dict = {k: [] for k in first_frame.keys()}

    for i in range(start_frame, end_frame):
        frame = dataset[i]
        for k, v in frame.items():
            episode_dict[k].append(v)

    # Stack into arrays/tensors
    # dataset[i] returns torch tensors or numpy arrays depending on backend, usually tensors
    collapsed_dict = {}
    for k, v_list in episode_dict.items():
        if len(v_list) > 0:
            if isinstance(v_list[0], torch.Tensor):
                collapsed_dict[k] = torch.stack(v_list)
            else:
                collapsed_dict[k] = np.stack(v_list)
        else:
            collapsed_dict[k] = np.array([])

    return collapsed_dict


def generate_stats_if_missing(dataset_root: Path):
    """
    Check if meta/episodes_stats.jsonl exists. If not, generate it and meta/stats.json
    using LeRobot compute_stats utilities.
    """
    episodes_stats_path = dataset_root / "meta/episodes_stats.jsonl"
    stats_path = dataset_root / "meta/stats.json"

    if episodes_stats_path.exists() and stats_path.exists():
        print(f"Stats files found in {dataset_root / 'meta'}, skipping generation.")
        return

    print("Missing stats files! Generating them now (this may take a while)...")

    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found in {dataset_root}")

    with open(info_path) as f:
        info = json.load(f)
    features = info.get("features", {})

    if not features:
        raise ValueError("No 'features' key found in info.json")

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} episodes to process for stats generation.")

    all_stats = []
    episode_stats_list = []

    for pq_path in tqdm(parquet_files, desc="Computing stats"):
        # Load episode
        df = pd.read_parquet(pq_path)

        # Convert to dictionary of arrays
        episode_data = {}
        for col in df.columns:
            if col not in features:
                continue

            val = df[col].values
            if val.dtype == object and len(val) > 0 and isinstance(val[0], (list, tuple, np.ndarray)):
                with suppress(Exception):
                    val = np.stack(val)
            episode_data[col] = val

        try:
            stats = compute_episode_stats(episode_data, features)
            all_stats.append(stats)

            # Prepare for per-episode jsonl
            stats_with_index = stats.copy()
            episode_index = int(pq_path.stem.split("_")[-1])
            stats_with_index["episode_index"] = episode_index
            episode_stats_list.append(stats_with_index)

        except Exception as e:
            print(f"Error computing stats for {pq_path}: {e}")
            raise e

    print("Aggregating statistics...")
    aggregated = aggregate_stats(all_stats)

    def numpy_converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save aggregated stats
    with open(stats_path, "w") as f:
        json.dump(aggregated, f, indent=4, default=numpy_converter)
    print(f"Saved aggregated stats to {stats_path}")

    # Save per-episode stats
    with open(episodes_stats_path, "w") as f:
        for stat in episode_stats_list:
            clean_stat = json.loads(json.dumps(stat, default=numpy_converter))
            output_item = {
                "episode_index": clean_stat["episode_index"],
                "stats": {k: v for k, v in clean_stat.items() if k != "episode_index"},
            }
            f.write(json.dumps(output_item) + "\n")

    print(f"Saved per-episode stats to {episodes_stats_path}")


def convert_lerobot_folder_to_hdf5(
    dataset_directory: str,
    output_hdf5_path: str,
    include_videos: bool,
    compression: str | None,
    compression_level: int,
    sort_column: str | None,
    convert_to_isaaclab_radians: bool,
    lerobot_action_column_name: str,
    lerobot_state_column_name: str,
):
    dataset_root = Path(dataset_directory).expanduser().resolve()

    # Check and generate stats if missing
    generate_stats_if_missing(dataset_root)

    output_file_path = Path(output_hdf5_path).expanduser().resolve()
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    videos_directory = dataset_root / "videos"
    meta_directory = dataset_root / "meta"

    with h5py.File(output_file_path, "w") as output_h5:
        output_h5.attrs["source_dir"] = str(dataset_root)
        output_h5.attrs["layout_version"] = "lerobot_v2.1_episode_based__optionally_converted_to_isaaclab_radians"
        output_h5.attrs["created_at"] = dt.datetime.now().isoformat()
        output_h5.attrs["convert_to_isaaclab_radians"] = bool(convert_to_isaaclab_radians)
        # Note: sort_column is not used when using LeRobotDataset as it handles loading
        output_h5.attrs["sort_column_requested"] = "" if sort_column is None else str(sort_column)
        output_h5.attrs["lerobot_action_column_name"] = lerobot_action_column_name
        output_h5.attrs["lerobot_state_column_name"] = lerobot_state_column_name

        # Write meta files
        if meta_directory.exists():
            meta_group = output_h5.require_group("meta")
            for meta_file_path in sorted(meta_directory.glob("*")):
                if not meta_file_path.is_file():
                    continue
                try:
                    text = meta_file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = meta_file_path.read_bytes().decode("utf-8", errors="ignore")
                write_text_dataset(meta_group, meta_file_path.name, text)

        # Initialize LeRobotDataset
        # Note: repo_id is required but for local usage we can just pass the name.
        # root must be the direct path to the dataset folder.
        dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)

        # Convert episodes using LeRobotDataset
        for episode_idx in tqdm(range(dataset.num_episodes), desc="Converting episodes"):
            # Load episode data manually
            episode_data = load_episode_data(dataset, episode_idx)
            episode_name = f"episode_{episode_idx:06d}"

            # NOTE: Flattened structure to match isaaclab2lerobot expectations: /data/<episode_name>
            episode_group = output_h5.require_group(f"data/{episode_name}")
            # episode_group.attrs["parquet_relpath"] = str(parquet_path.relative_to(dataset_root)) # Not applicable

            # Use 'index' column for length if available, else use first column length
            num_frames = len(episode_data[next(iter(episode_data))])
            episode_group.attrs["num_frames"] = int(num_frames)
            # episode_group.attrs["sorted_by"] = used_sort_column # LeRobotDataset returns sorted frames

            # Write each column
            for column_name, column_tensor in episode_data.items():
                if isinstance(column_tensor, torch.Tensor):
                    numpy_array = column_tensor.numpy()
                else:
                    numpy_array = np.array(column_tensor)

                # Normalize float dtype if possible
                if numpy_array.dtype == np.float64:
                    numpy_array = numpy_array.astype(np.float32)

                # If enabled: convert action/state columns from LeRobot normalized degrees -> IsaacLab radians
                if convert_to_isaaclab_radians and numpy_array.ndim == 2:
                    if column_name in (lerobot_action_column_name, lerobot_state_column_name):
                        numpy_array = denormalize_lerobot_to_isaaclab_radians(numpy_array)

                safe_dataset_name = column_name.replace("/", "_")
                write_numpy_dataset(
                    h5_group=episode_group,
                    dataset_name=safe_dataset_name,
                    array=numpy_array,
                    compression=compression,
                    compression_level=compression_level if compression else None,
                )

                # Note: `note` logic from stack_object_column_to_numpy is skipped as LeRobotDataset returns tensors

        # Optionally store mp4 bytes in HDF5
        if include_videos and videos_directory.exists():
            video_files = sorted(videos_directory.glob("chunk-*/**/episode_*.mp4"))
            videos_group = output_h5.require_group("videos")

            for video_path in tqdm(video_files, desc="Packing videos into HDF5"):
                relative_video_path = video_path.relative_to(dataset_root)
                parts = relative_video_path.parts
                if len(parts) < 4:
                    continue

                _, chunk_name, camera_name = parts[0], parts[1], parts[2]
                episode_filename = parts[-1]
                episode_stem = Path(episode_filename).stem

                camera_group = videos_group.require_group(f"{chunk_name}/{camera_name}")
                dataset_name = f"{episode_stem}.mp4"

                if dataset_name in camera_group:
                    del camera_group[dataset_name]

                video_bytes = read_file_as_uint8_bytes(video_path)
                camera_group.create_dataset(
                    dataset_name,
                    data=video_bytes,
                    dtype=np.uint8,
                    compression=compression,
                    compression_opts=compression_level if compression else None,
                    shuffle=True if compression else None,
                )
                camera_group[dataset_name].attrs["video_relpath"] = str(relative_video_path)

    print(f"[OK] Wrote HDF5 to: {output_file_path}")


def main():
    argument_parser = argparse.ArgumentParser(
        description="Convert local LeRobot dataset folder (v2.1) to a single HDF5 file."
    )
    argument_parser.add_argument("--lerobot_dir", type=str, required=True, help="Local LeRobot dataset folder path")
    argument_parser.add_argument("--output_hdf5", type=str, required=True, help="Output HDF5 file path")

    argument_parser.add_argument(
        "--include_videos",
        action="store_true",
        help="Store mp4 bytes into HDF5 (may be large)",
    )
    argument_parser.add_argument(
        "--no_compression",
        action="store_true",
        help="Disable HDF5 compression",
    )
    argument_parser.add_argument(
        "--compression_level",
        type=int,
        default=4,
        help="gzip compression level (1-9)",
    )
    argument_parser.add_argument(
        "--sort_column",
        type=str,
        default=None,
        help=(
            "Sort frames inside each episode by this column if present. "
            "If not provided, tries: index > frame_index > timestamp."
        ),
    )
    argument_parser.add_argument(
        "--no_convert_to_isaaclab_radians",
        action="store_true",
        help="Disable conversion of LeRobot normalized degrees to IsaacLab radians (default: conversion is ENABLED).",
    )
    argument_parser.add_argument(
        "--lerobot_action_column_name",
        type=str,
        default="action",
        help="Column name in parquet for action vectors (default: action)",
    )
    argument_parser.add_argument(
        "--lerobot_state_column_name",
        type=str,
        default="observation.state",
        help="Column name in parquet for state vectors (default: observation.state)",
    )

    args = argument_parser.parse_args()

    compression = None if args.no_compression else "gzip"

    convert_lerobot_folder_to_hdf5(
        dataset_directory=args.lerobot_dir,
        output_hdf5_path=args.output_hdf5,
        include_videos=args.include_videos,
        compression=compression,
        compression_level=args.compression_level,
        sort_column=args.sort_column,
        convert_to_isaaclab_radians=not args.no_convert_to_isaaclab_radians,
        lerobot_action_column_name=args.lerobot_action_column_name,
        lerobot_state_column_name=args.lerobot_state_column_name,
    )


if __name__ == "__main__":
    main()
