"""
Convert a local LeRobot dataset folder (v2.1 episode-based layout) to a single HDF5 file.
Extracts selected columns (default: 'action'), applies denormalization to 'action'.
"""

import argparse
import datetime as dt
import json
import os
from contextlib import suppress
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Disable HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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
    LeRobot normalized degrees -> IsaacLab joint limits (degrees) -> radians.
    """
    joint_values_lerobot = np.asarray(joint_values_lerobot, dtype=np.float32)

    if joint_values_lerobot.ndim != 2:
        raise ValueError(f"Expected 2D array (T,D), got {joint_values_lerobot.shape}")

    dimension = joint_values_lerobot.shape[1]
    if dimension < 6:
        raise ValueError(f"Expected D>=6, got D={dimension}")

    result_deg = joint_values_lerobot.copy()

    # Map first 6 joints (or 6+6 for bimanual)
    if dimension == 12:
        # bimanual
        for arm_offset in (0, 6):
            for joint_index in range(6):
                isa_min, isa_max = ISAACLAB_LIMITS_DEG[joint_index]
                le_min, le_max = LEROBOT_LIMITS_DEG[joint_index]
                column_index = arm_offset + joint_index
                result_deg[:, column_index] = (result_deg[:, column_index] - le_min) / (le_max - le_min) * (
                    isa_max - isa_min
                ) + isa_min
    else:
        # single arm
        for joint_index in range(6):
            isa_min, isa_max = ISAACLAB_LIMITS_DEG[joint_index]
            le_min, le_max = LEROBOT_LIMITS_DEG[joint_index]
            result_deg[:, joint_index] = (result_deg[:, joint_index] - le_min) / (le_max - le_min) * (
                isa_max - isa_min
            ) + isa_min

    return result_deg * (np.pi / 180.0)


def generate_stats_if_missing(dataset_root: Path):
    """
    Check if meta/episodes_stats.jsonl exists. If not, generate all stats.
    """
    episodes_stats_path = dataset_root / "meta/episodes_stats.jsonl"
    stats_path = dataset_root / "meta/stats.json"

    if episodes_stats_path.exists() and stats_path.exists():
        print(f"Stats files found in {dataset_root / 'meta'}, skipping generation.")
        return

    print("Missing stats files! Generating them now...")

    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found in {dataset_root}")

    with open(info_path) as info_file:
        info = json.load(info_file)
    features = info.get("features", {})

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    all_stats = []
    episode_stats_list = []

    for parquet_path in tqdm(parquet_files, desc="Computing stats"):
        dataframe = pd.read_parquet(parquet_path)
        episode_data = {}
        for column_name in dataframe.columns:
            if column_name not in features:
                continue
            column_values = dataframe[column_name].values
            # Handle object columns (lists/arrays)
            if (
                column_values.dtype == object
                and len(column_values) > 0
                and isinstance(column_values[0], (list, tuple, np.ndarray))
            ):
                with suppress(Exception):
                    column_values = np.stack(column_values)
            episode_data[column_name] = column_values

        try:
            stats = compute_episode_stats(episode_data, features)
            all_stats.append(stats)

            stats_with_index = stats.copy()
            stats_with_index["episode_index"] = int(parquet_path.stem.split("_")[-1])
            episode_stats_list.append(stats_with_index)
        except Exception as e:
            print(f"Error computing stats for {parquet_path}: {e}")
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

    with open(stats_path, "w") as stats_file:
        json.dump(aggregated, stats_file, indent=4, default=numpy_converter)

    with open(episodes_stats_path, "w") as episodes_stats_file:
        for stat in episode_stats_list:
            clean_stat = json.loads(json.dumps(stat, default=numpy_converter))
            output_item = {
                "episode_index": clean_stat["episode_index"],
                "stats": {k: v for k, v in clean_stat.items() if k != "episode_index"},
            }
            episodes_stats_file.write(json.dumps(output_item) + "\n")

    print(f"Stats generated at {episodes_stats_path}")


def convert_lerobot_folder_to_hdf5(
    lerobot_dir: str,
    output_hdf5_path: str,
    column_keys: list[str] = ["action"],
    list_available_keys: bool = False,
    use_all_keys: bool = False,
):
    dataset_root = Path(lerobot_dir).expanduser().resolve()
    output_path = Path(output_hdf5_path).expanduser().resolve()

    # 1. Ensure stats exist
    generate_stats_if_missing(dataset_root)

    print(f"Loading LeRobotDataset from {dataset_root}...")
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)

    if use_all_keys:
        if dataset.num_episodes > 0:
            first_idx = dataset.episode_data_index["from"][0].item()
            # Get keys from the first frame
            column_keys = list(dataset[first_idx].keys())
            print(f"[Info] Using all available keys: {column_keys}")
        else:
            print("[Warning] Dataset appears empty, cannot determine keys.")

    if list_available_keys:
        print("\n[Available Keys in Dataset]")
        if dataset.num_episodes > 0:
            # Peek at first frame
            first_idx = dataset.episode_data_index["from"][0].item()
            keys = list(dataset[first_idx].keys())
            for k in keys:
                print(f"  - {k}")
        else:
            print("  (Dataset is empty)")
        return

    # 2. Setup output HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as output_hdf5_file:
        output_hdf5_file.attrs["source_dir"] = str(dataset_root)
        output_hdf5_file.attrs["created_at"] = dt.datetime.now().isoformat()
        output_hdf5_file.attrs["convert_to_isaaclab_radians"] = True

        # Write meta files (optional but good for provenance)
        meta_group = output_hdf5_file.create_group("meta")
        if (dataset_root / "meta").exists():
            for meta_file_path in (dataset_root / "meta").glob("*"):
                if meta_file_path.is_file():
                    try:
                        content = meta_file_path.read_text(encoding="utf-8", errors="ignore")
                        meta_group.create_dataset(
                            meta_file_path.name, data=content, dtype=h5py.string_dtype(encoding="utf-8")
                        )
                    except Exception:
                        pass

        # 3. Iterate episodes and save selected columns
        for episode_index in tqdm(range(dataset.num_episodes), desc="Converting to HDF5"):
            # Load frames
            start_index = dataset.episode_data_index["from"][episode_index].item()
            end_index = dataset.episode_data_index["to"][episode_index].item()

            frames = [dataset[i] for i in range(start_index, end_index)]
            if not frames:
                continue

            # Create Group with new naming convention
            episode_group_name = f"data/demo_{episode_index}"
            episode_group = output_hdf5_file.require_group(episode_group_name)

            # Set frames count (based on first requested key found, or total frames)
            episode_group.attrs["num_frames"] = len(frames)

            for key in column_keys:
                # Extract specific column
                try:
                    values_list = [frame[key] for frame in frames]
                except KeyError:
                    print(f"Warning: Key '{key}' not found in episode {episode_index}, skipping.")
                    continue

                # Stack
                if isinstance(values_list[0], torch.Tensor):
                    values_array = torch.stack(values_list).numpy()
                else:
                    values_array = np.array(values_list)

                if key == "action":
                    values_array = denormalize_lerobot_to_isaaclab_radians(values_array)

                # Save compressed
                # Sanitize key for HDF5 dataset name
                safe_key = key.replace("/", "_")

                # Handle string data types (e.g. 'task')
                if values_array.dtype.kind in ("U", "S"):
                    episode_group.create_dataset(
                        safe_key,
                        data=values_array.astype("S"),  # Convert to fixed-length bytes (simplest for compatibility)
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    episode_group.create_dataset(
                        safe_key,
                        data=values_array,
                        compression="gzip",
                        compression_opts=4,
                    )

    print(f"[OK] Wrote HDF5 to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compact LeRobot to HDF5 converter")
    parser.add_argument("--lerobot_dir", type=str, required=True, help="Path to local LeRobot dataset")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path to output HDF5")
    parser.add_argument(
        "--column_keys", type=str, nargs="+", default=["action"], help="Column names to extract (default: action)"
    )
    parser.add_argument("--all_keys", action="store_true", help="Extract all available keys (overrides --column_keys)")
    parser.add_argument("--list_keys", action="store_true", help="List available column keys in the dataset and exit")

    args = parser.parse_args()

    convert_lerobot_folder_to_hdf5(
        lerobot_dir=args.lerobot_dir,
        output_hdf5_path=args.output_hdf5,
        column_keys=args.column_keys,
        list_available_keys=args.list_keys,
        use_all_keys=args.all_keys,
    )


if __name__ == "__main__":
    main()
