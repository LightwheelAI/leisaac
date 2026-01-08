#!/usr/bin/env python3

"""
Convert IDM parquet (LeRobot-style rows) + source LeIsaac HDF5 (initial_state/template)
to a new replayable LeIsaac HDF5.

Input parquet row example:
{
  "observation.state":[...6...],
  "action":[...6...],
  "timestamp":0,
  "episode_index":0,
  "index":0
}

Source HDF5 episode structure (example):
/actions (T,6)
/initial_state/...
/obs/actions (T,6)
/processed_actions (T,6)
/states/articulation/robot/joint_position (T,6)
...

We will:
- Copy root metadata groups (everything except /data) from source HDF5.
- For each episode:
  - Copy initial_state + attrs from a template episode in source HDF5
  - Write /actions, /obs/actions, /processed_actions from parquet "action"
  - Optionally write /states/.../joint_position from parquet "observation.state"
  - Optionally write /timestamps from parquet "timestamp"
  - Optionally copy template obs/states arrays (including images) from source HDF5(can be huge)
"""

import argparse
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------------
# Denormalize function (same as isaaclab2lerobot)
# ------------------------

ISAACLAB_LIMITS = [(-110.0, 110.0), (-100.0, 100.0), (-100.0, 90.0), (-95.0, 95.0), (-160.0, 160.0), (-10.0, 100.0)]
LEROBOT_LIMITS = [(-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (0.0, 100.0)]


def denormalize_lerobot_to_isaaclab_radians(joint_pos: np.ndarray) -> np.ndarray:
    """
    LeRobot normalized degrees in roughly [-100,100] -> IsaacLab joint limits (degrees) -> radians.
    joint_pos: (T,6)
    """
    result = joint_pos.astype(np.float32).copy()
    for i in range(6):
        isa_min, isa_max = ISAACLAB_LIMITS[i]
        le_min, le_max = LEROBOT_LIMITS[i]
        isa_range = isa_max - isa_min
        le_range = le_max - le_min
        result[:, i] = (result[:, i] - le_min) / le_range * isa_range + isa_min
    # degrees -> radians
    result = result * (np.pi / 180.0)
    return result


# ------------------------
# HDF5 copy helpers
# ------------------------


def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def copy_group_recursive(src: h5py.Group, dst: h5py.Group, skip: list[str] | None = None):
    """
    Recursively copy a group, skipping relative paths in `skip`.
    """
    skip = set(skip or [])

    def _rec(s: h5py.Group | h5py.File, d: h5py.Group, rel: str):
        copy_attrs(s, d)
        for name, item in s.items():
            child_rel = f"{rel}/{name}" if rel else name
            if child_rel in skip:
                continue
            if isinstance(item, h5py.Dataset):
                ds = d.create_dataset(name, data=item[()], compression=item.compression)
                for ak, av in item.attrs.items():
                    ds.attrs[ak] = av
            else:
                g = d.create_group(name)
                _rec(item, g, child_rel)

    _rec(src, dst, "")


def ensure_group(root: h5py.Group, path: str) -> h5py.Group:
    """
    Ensure group path exists under root. Return the group.
    """
    path = path.strip("/")
    cur = root
    if not path:
        return cur
    for part in path.split("/"):
        if part not in cur:
            cur = cur.create_group(part)
        else:
            cur = cur[part]
            if not isinstance(cur, h5py.Group):
                raise ValueError(f"Path conflict: {path} has non-group at {part}")
    return cur


def write_dataset(root: h5py.Group, path: str, data: np.ndarray, compression: str | None = None):
    """
    Overwrite dataset at path with given data.
    """
    path = path.strip("/")
    parent_path, name = path.rsplit("/", 1) if "/" in path else ("", path)
    parent = ensure_group(root, parent_path)
    if name in parent:
        del parent[name]
    parent.create_dataset(name, data=data, compression=compression)


# ------------------------
# Parquet loading
# ------------------------

_EP_RE = re.compile(r"episode_(\d+)\.parquet$")


def list_episode_parquet_files(parquet_dir: str) -> list[tuple[int, Path]]:
    """Return [(episode_id_from_filename, path), ...] sorted by episode_id."""
    p = Path(parquet_dir)
    if not p.is_dir():
        raise ValueError(f"--parquet must be a directory when using filename mapping: {parquet_dir}")
    files = sorted(p.glob("**/episode_*.parquet"))
    items: list[tuple[int, Path]] = []
    for f in files:
        m = _EP_RE.search(f.name)
        if not m:
            continue
        items.append((int(m.group(1)), f))
    if not items:
        raise FileNotFoundError(f"No episode_*.parquet found under: {parquet_dir}")
    items.sort(key=lambda x: x[0])
    return items


def load_parquet_input(parquet_path_or_dir: str) -> pd.DataFrame:
    p = Path(parquet_path_or_dir)
    if p.is_dir():
        files = sorted(
            p.glob("**/*.parquet"),
            key=lambda x: (
                int(re.search(r"episode_(\d+)", x.name).group(1)) if re.search(r"episode_(\d+)", x.name) else 0
            ),
        )
        if not files:
            raise FileNotFoundError(f"No parquet files found in dir: {parquet_path_or_dir}")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        if not p.exists():
            raise FileNotFoundError(parquet_path_or_dir)
        df = pd.read_parquet(p)
    return df


def stack_vector_column(df: pd.DataFrame, col: str) -> np.ndarray:
    # column entries are list/np.ndarray per row
    return np.stack(df[col].to_numpy())


# ------------------------
# Main conversion
# ------------------------


def convert(  # noqa: C901
    parquet_path_or_dir: str,
    src_hdf5: str,
    out_hdf5: str,
    overwrite: bool,
    episode_col: str,
    sort_col: str | None,
    action_col: str,
    state_col: str,
    timestamp_col: str | None,
    write_states: bool,
    keep_template_obs: bool,
):
    if os.path.exists(out_hdf5):
        if overwrite:
            os.remove(out_hdf5)
        else:
            raise FileExistsError(f"Output exists: {out_hdf5} (use --overwrite)")

    # Use filename-based mapping when input is a directory containing episode_*.parquet
    parquet_items = list_episode_parquet_files(parquet_path_or_dir) if Path(parquet_path_or_dir).is_dir() else None

    if parquet_items is None:
        # fallback: single parquet file mode (still uses episode_index)
        df = load_parquet_input(parquet_path_or_dir)

        for need in [episode_col, action_col]:
            if need not in df.columns:
                raise ValueError(f"Missing required column '{need}' in parquet. columns={list(df.columns)}")

        if write_states and state_col not in df.columns:
            raise ValueError(f"--write_states enabled but missing column '{state_col}'")

        if sort_col and sort_col not in df.columns:
            raise ValueError(f"--sort_col '{sort_col}' not in columns")

        if timestamp_col and timestamp_col not in df.columns:
            raise ValueError(f"--timestamp_col '{timestamp_col}' not in columns")

        ep_ids = sorted(df[episode_col].unique().tolist())
        print(f"[INFO] parquet episodes (from column {episode_col}): {len(ep_ids)}")
    else:
        print(f"[INFO] parquet files (from filename): {len(parquet_items)}")

    with h5py.File(src_hdf5, "r") as f_src, h5py.File(out_hdf5, "w") as f_out:
        if "data" not in f_src:
            raise ValueError("Source HDF5 missing /data")

        # copy root metadata except /data
        for k, item in f_src.items():
            if k == "data":
                continue
            if isinstance(item, h5py.Dataset):
                ds = f_out.create_dataset(k, data=item[()], compression=item.compression)
                for ak, av in item.attrs.items():
                    ds.attrs[ak] = av
            else:
                g = f_out.create_group(k)
                copy_group_recursive(item, g)

        g_data_out = f_out.create_group("data")
        g_data_src = f_src["data"]
        copy_attrs(g_data_src, g_data_out)

        src_ep_names = list(g_data_src.keys())
        if not src_ep_names:
            raise ValueError("Source HDF5 has zero episodes under /data")
        template_name = src_ep_names[0]

        # We will copy:
        # - attrs
        # - initial_state group
        # optionally: obs/states skeleton (but not the big image arrays unless user wants to keep)
        # Note: copying images can explode file size; default keep_template_obs=False.
        base_skip = []
        if not keep_template_obs:
            # skip heavy video/images and other per-step arrays from template
            # we'll only write the minimum required datasets.
            base_skip += [
                "obs/front",
                "obs/ee_frame_state",
                "obs/joint_pos",
                "obs/joint_pos_rel",
                "obs/joint_pos_target",
                "obs/joint_vel",
                "obs/joint_vel_rel",
                "states",
                "actions",
                "obs/actions",
                "processed_actions",
            ]

        if parquet_items is None:
            # old behavior: split by episode_col inside one big df
            iterator = [(int(ep_id), None) for ep_id in ep_ids]
        else:
            # new behavior: each file defines one episode, use filename episode id
            iterator = parquet_items  # [(ep_id, path), ...]

        for ep_id, pq_path in tqdm(iterator, desc="Writing episodes"):
            if pq_path is None:
                dfe = df[df[episode_col] == ep_id].copy()
            else:
                dfe = pd.read_parquet(pq_path)

            # basic column checks (per-file for filename mode)
            if action_col not in dfe.columns:
                raise ValueError(f"Missing required column '{action_col}' in parquet for ep={ep_id}: {pq_path}")
            if write_states and state_col not in dfe.columns:
                raise ValueError(f"--write_states enabled but missing '{state_col}' for ep={ep_id}: {pq_path}")
            if sort_col and sort_col not in dfe.columns:
                raise ValueError(f"--sort_col '{sort_col}' not in parquet columns for ep={ep_id}: {pq_path}")
            if timestamp_col and timestamp_col not in dfe.columns:
                raise ValueError(f"--timestamp_col '{timestamp_col}' not in parquet columns for ep={ep_id}: {pq_path}")

            if sort_col:
                dfe = dfe.sort_values(sort_col, ascending=True)

            # actions (T,6) in lerobot normalized degrees
            act_raw = stack_vector_column(dfe, action_col).astype(np.float32)
            if act_raw.ndim != 2 or act_raw.shape[1] != 6:
                raise ValueError(f"Action shape expected (T,6), got {act_raw.shape} for ep={ep_id}")
            act = denormalize_lerobot_to_isaaclab_radians(act_raw)

            # states (T,6) optional
            st = None
            if write_states:
                st_raw = stack_vector_column(dfe, state_col).astype(np.float32)
                if st_raw.ndim != 2 or st_raw.shape[1] != 6:
                    raise ValueError(f"State shape expected (T,6), got {st_raw.shape} for ep={ep_id}")
                st = denormalize_lerobot_to_isaaclab_radians(st_raw)

            ts = None
            if timestamp_col:
                ts = dfe[timestamp_col].to_numpy(dtype=np.int64)

            # choose a source episode to copy initial_state from (now: use ep_id from filename)
            src_name = f"demo_{int(ep_id)}"
            if src_name not in g_data_src:
                # fallback map by index
                if int(ep_id) < len(src_ep_names):
                    src_name = src_ep_names[int(ep_id)]
                else:
                    src_name = template_name

            g_src_ep = g_data_src[src_name]
            dst_name = f"demo_{int(ep_id)}"
            g_dst_ep = g_data_out.create_group(dst_name)

            # copy template stuff (attrs + initial_state + maybe skeleton)
            copy_attrs(g_src_ep, g_dst_ep)
            if "initial_state" in g_src_ep:
                g_init = g_dst_ep.create_group("initial_state")
                copy_group_recursive(g_src_ep["initial_state"], g_init)
            else:
                raise ValueError(f"Source episode {src_name} missing /initial_state")

            for grp_name in ["obs", "states"]:
                if grp_name in g_src_ep and keep_template_obs:
                    g = g_dst_ep.create_group(grp_name)
                    copy_group_recursive(g_src_ep[grp_name], g)

            # Write actions into required places
            write_dataset(g_dst_ep, "actions", act, compression=None)
            write_dataset(g_dst_ep, "obs/actions", act, compression=None)
            write_dataset(g_dst_ep, "processed_actions", act, compression=None)

            if write_states and st is not None:
                write_dataset(g_dst_ep, "states/articulation/robot/joint_position", st, compression=None)

            if ts is not None:
                write_dataset(g_dst_ep, "timestamps", ts, compression=None)

            g_dst_ep.attrs["num_steps"] = int(act.shape[0])
            g_dst_ep.attrs["action_dim"] = int(act.shape[1])
            g_dst_ep.attrs["converted_from_src_episode"] = src_name
            g_dst_ep.attrs["idm_source"] = str(Path(parquet_path_or_dir).resolve())
            if pq_path is not None:
                g_dst_ep.attrs["idm_parquet_file"] = pq_path.name

        print(f"[DONE] Output written: {out_hdf5}")
        print("Run replay with:")
        print(f"  --dataset_file {out_hdf5}")


def build_parser():
    p = argparse.ArgumentParser("IDM parquet -> LeIsaac HDF5 converter")
    p.add_argument("--parquet", type=str, required=True, help="Parquet file OR directory containing parquet(s).")
    p.add_argument("--src_hdf5", type=str, required=True, help="Source teleop HDF5 for initial_state/template.")
    p.add_argument("--out_hdf5", type=str, required=True, help="Output HDF5 for replay.")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--episode_col", type=str, default="episode_index")
    p.add_argument("--sort_col", type=str, default="index", help="Order within episode. default: index")
    p.add_argument("--action_col", type=str, default="action")
    p.add_argument("--state_col", type=str, default="observation.state")
    p.add_argument("--timestamp_col", type=str, default="timestamp")

    p.add_argument(
        "--write_states",
        action="store_true",
        help="Also write /states/articulation/robot/joint_position from observation.state",
    )
    p.add_argument(
        "--keep_template_obs",
        action="store_true",
        help="Copy template obs/states arrays (including images) from source (can be huge). Default off.",
    )
    return p


def main():
    args = build_parser().parse_args()
    convert(
        parquet_path_or_dir=args.parquet,
        src_hdf5=args.src_hdf5,
        out_hdf5=args.out_hdf5,
        overwrite=args.overwrite,
        episode_col=args.episode_col,
        sort_col=args.sort_col,
        action_col=args.action_col,
        state_col=args.state_col,
        timestamp_col=args.timestamp_col,
        write_states=args.write_states,
        keep_template_obs=args.keep_template_obs,
    )


if __name__ == "__main__":
    main()
