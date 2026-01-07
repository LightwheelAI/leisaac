#!/usr/bin/env python3
"""
Convert cosmos2.5 outputs (+ optional LeRobot meta) to convert_directory output format.

Input:
  cosmos_dir/
    ├── *.mp4
    ├── *.json(same stem as mp4)
    └── ...

Optional:
  lerobot_dir/
    └── meta/
        ├── tasks.jsonl
        └── episodes.jsonl

Output:
  output_dir/
    ├── <TaskName>/
    │   ├── 0.mp4
    │   ├── 1.mp4
    │   └── ...
    └── ...

Rule:
  - If json has "prompt": use it as task name (sanitize to dir)
  - Else: fallback to LeRobot meta (requires --lerobot_dir)
    - If lerobot_dir not provided: print warning and skip that sample
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def sanitize_task_name(task: str) -> str:
    """
    Convert task description to valid directory name.
    Example: "Lift the red cube up." -> "Lift_the_red_cube_up"
    """
    task = task.strip()
    for ch in [".", ",", "!", "?", ":", ";", "\"", "'"]:
        task = task.replace(ch, "")
    task = task.replace(" ", "_")
    task = "_".join(filter(None, task.split("_")))
    return task


def load_tasks(tasks_file: Path) -> Dict[int, str]:
    """Load task definitions. Returns {task_index: task_string}"""
    tasks: Dict[int, str] = {}
    with open(tasks_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            tasks[int(data["task_index"])] = data["task"]
    return tasks


def load_episodes(episodes_file: Path) -> List[dict]:
    """Load episode information. Returns list of dict (episode_index, tasks, length, ...)"""
    episodes: List[dict] = []
    with open(episodes_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            episodes.append(json.loads(line))
    return episodes


def build_episode_to_task_map(lerobot_dir: Path) -> Dict[int, str]:
    """
    Build mapping: episode_index -> task_string (first task in episode['tasks'])
    """
    meta_dir = lerobot_dir / "meta"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"

    if not tasks_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"LeRobot meta files not found under: {meta_dir}")

    _tasks = load_tasks(tasks_path)  # not strictly needed, but useful if episodes store indices in some setups
    episodes = load_episodes(episodes_path)

    ep2task: Dict[int, str] = {}
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        ep_tasks = ep.get("tasks", [])
        if not ep_tasks:
            continue

        # In many LeRobot datasets, ep["tasks"] stores task strings already.
        # If it stores indices, you can extend here. We'll support both:
        t0 = ep_tasks[0]
        if isinstance(t0, int):
            task_str = _tasks.get(int(t0), str(t0))
        else:
            task_str = str(t0)

        ep2task[ep_idx] = task_str

    return ep2task


def read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read json: {path} ({e})")
        return None


def parse_episode_index_from_stem(stem: str) -> Optional[int]:
    """
    Try to extract episode index from filename stem.
    Examples it can handle:
      - episode_000123
      - ..._000123
      - 000123
    If cannot parse, return None.
    """
    # Most common: "episode_000123"
    if stem.startswith("episode_"):
        tail = stem[len("episode_") :]
        if tail.isdigit():
            return int(tail)

    # Try last underscore chunk
    parts = stem.split("_")
    for candidate in reversed(parts):
        if candidate.isdigit():
            return int(candidate)

    # Entire stem digits?
    if stem.isdigit():
        return int(stem)

    return None


def convert_cosmos_to_step2(
    cosmos_dir: Path,
    output_dir: Path,
    lerobot_dir: Optional[Path] = None,
    chunk_missing_prompt_policy: str = "skip",
):
    """
    Convert cosmos2.5 outputs to step2 format.

    chunk_missing_prompt_policy:
      - "skip": if no prompt and no lerobot_dir mapping, skip the sample
    """
    cosmos_dir = cosmos_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ep2task: Dict[int, str] = {}
    if lerobot_dir is not None:
        ep2task = build_episode_to_task_map(lerobot_dir.resolve())

    # Collect pairs (json, mp4) by stem
    json_files = sorted(cosmos_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in cosmos_dir: {cosmos_dir}")

    # task_name -> list of source mp4 paths (ordered)
    task_to_videos: Dict[str, List[Path]] = defaultdict(list)
    skipped: List[Tuple[Path, str]] = []

    for jpath in json_files:
        stem = jpath.stem
        mpath = cosmos_dir / f"{stem}.mp4"
        if not mpath.exists():
            print(f"[WARN] Missing mp4 for json: {jpath.name} -> expected {mpath.name}, skip.")
            skipped.append((jpath, "missing_mp4"))
            continue

        data = read_json(jpath)
        if data is None:
            skipped.append((jpath, "bad_json"))
            continue

        prompt = data.get("prompt", None)
        if isinstance(prompt, str) and prompt.strip():
            task_str = prompt.strip()
            task_dir_name = sanitize_task_name(task_str)
            task_to_videos[task_dir_name].append(mpath)
            continue

        # No prompt -> fallback to lerobot meta mapping
        if not ep2task:
            print(
                f"[WARN] {jpath.name} has no 'prompt'. "
                f"You didn't provide --lerobot_dir (or mapping is empty), cannot infer task. Skipping."
            )
            skipped.append((jpath, "no_prompt_no_lerobot"))
            continue

        ep_idx = parse_episode_index_from_stem(stem)
        if ep_idx is None:
            print(
                f"[WARN] {jpath.name} has no 'prompt' and episode index cannot be parsed from name '{stem}'. Skipping."
            )
            skipped.append((jpath, "no_prompt_cannot_parse_episode"))
            continue

        task_str = ep2task.get(ep_idx)
        if not task_str:
            print(
                f"[WARN] {jpath.name} has no 'prompt'. Parsed episode_index={ep_idx}, "
                f"but it's not found in lerobot meta. Skipping."
            )
            skipped.append((jpath, "episode_not_in_meta"))
            continue

        task_dir_name = sanitize_task_name(task_str)
        task_to_videos[task_dir_name].append(mpath)

    # Copy into output folders with sequential numbering per task
    total_copied = 0
    for task_dir_name, vids in sorted(task_to_videos.items(), key=lambda x: x[0]):
        dst_task_dir = output_dir / task_dir_name
        dst_task_dir.mkdir(parents=True, exist_ok=True)

        # keep deterministic order
        vids_sorted = sorted(vids, key=lambda p: p.name)
        for i, src in enumerate(vids_sorted):
            dst = dst_task_dir / f"{i}.mp4"
            shutil.copy2(src, dst)
            total_copied += 1

        print(f"[OK] Task '{task_dir_name}': {len(vids_sorted)} videos")

    print("\nConversion complete!")
    print(f"  Total videos copied: {total_copied}")
    print(f"  Output directory: {output_dir}")

    if skipped:
        print(f"\n[SUMMARY] Skipped {len(skipped)} samples:")
        # print a few for readability
        for p, reason in skipped[:20]:
            print(f"  - {p.name}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Convert cosmos2.5 outputs (+ optional LeRobot meta) to convert_directory output format"
    )
    parser.add_argument("--cosmos_dir", type=str, required=True, help="Cosmos output directory containing *.mp4 and *.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--lerobot_dir",
        type=str,
        default=None,
        help="Optional LeRobot dataset directory (only needed when json has no 'prompt')",
    )

    args = parser.parse_args()

    cosmos_dir = Path(args.cosmos_dir)
    output_dir = Path(args.output_dir)
    lerobot_dir = Path(args.lerobot_dir) if args.lerobot_dir else None

    convert_cosmos_to_step2(
        cosmos_dir=cosmos_dir,
        output_dir=output_dir,
        lerobot_dir=lerobot_dir,
    )


if __name__ == "__main__":
    main()
