"""
Hybrid Replay: Parquet actions/states + HDF5 initial states

This script replays recorded robot episodes by:
- Loading initial states from HDF5 datasets
- Loading actions or states from Parquet files
- Running replay inside IsaacLab environments
- Supporting parallel environments and success statistics
"""


import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

# ---------------------------------------------------------------------
# CLI arguments and IsaacLab app launcher
# ---------------------------------------------------------------------
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hybrid Replay for Leisaac")
parser.add_argument("--task", type=str, required=True, help="LeIsaac Task name,e.g. LeIsaac-SO101-LiftCube-v0")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--step_hz", type=int, default=60, help="Simulation frequency")
parser.add_argument("--parquet_dir", type=str, default="<path_to_output_lerobot/data/chunk-000>",
                    help="Directory containing Parquet episode files (expects episode_*.parquet)."
parser.add_argument("--hdf5_file", type=str, default="<path_to_dataset.hdf5>",
                    help="HDF5 dataset file containing per-episode initial states and metadata.")
parser.add_argument("--select_episodes", type=int, nargs="+", default=[], 
                    help="Episode indices to replay. If empty, replays all available parquet episodes.")
parser.add_argument("--task_type",type=str,default=None,help="Optional task type override. If not set, inferred from --task.")
parser.add_argument("--replay_mode",type=str,default="action",choices=["action", "state"],
                    help="Replay mode: 'action' replays action vectors; 'state' replays joint states.")

# Add IsaacLab standard launcher args (e.g., device/headless/experience settings).
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch IsaacLab app (must happen before importing certain sim-related modules).
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# ---------------------------------------------------------------------
# Imports (after app launch)
# ---------------------------------------------------------------------
import contextlib
import os
import time
import re
import gymnasium as gym
import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type


# ---------------------------------------------------------------------
# Episode container: parquet actions/states + HDF5 initial states
# ---------------------------------------------------------------------
class HybridEpisodeData:
    """
    Holds a single episode's replay payload:
      - initial_state: from HDF5
      - actions: from parquet column "action" (optional)
      - states: from parquet column "observation.state" (optional)

    Data is denormalized from LeRobot format into IsaacLab radians.
    """

    def __init__(self, actions_df: pd.DataFrame, initial_state: dict, device: str):
        self.device = device
        self.step_idx = 0
        self.initial_state = initial_state
        self.actions = None
        self.states = None

        # Parquet data uses normalized angles in [-100, 100].
        # We denormalize into IsaacLab joint limits and convert degrees -> radians.
        if "action" in actions_df.columns:
            actions_np = np.stack(actions_df["action"].to_numpy())
            actions_np = self._denormalize_joint_pos(actions_np)
            self.actions = torch.tensor(actions_np, device=device, dtype=torch.float32)

        if "observation.state" in actions_df.columns:
            states_np = np.stack(actions_df["observation.state"].to_numpy())
            states_np = self._denormalize_joint_pos(states_np)
            self.states = torch.tensor(states_np, device=device, dtype=torch.float32)

    @staticmethod
    def _denormalize_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
        """
        Denormalize: LeRobot normalized joint positions [-100, 100] -> IsaacLab joint limits (degrees),
        then convert degrees to radians.

        NOTE: The limits here must match the embodiment/task joint conventions.
        """
        ISAACLAB_LIMITS = [
            (-110.0, 110.0),
            (-100.0, 100.0),
            (-100.0, 90.0),
            (-95.0, 95.0),
            (-160.0, 160.0),
            (-10, 100.0),
        ]
        LEROBOT_LIMITS = [
            (-100, 100),
            (-100, 100),
            (-100, 100),
            (-100, 100),
            (-100, 100),
            (0, 100),
        ]

        result = joint_pos.copy()
        for i in range(6):
            isaaclab_min, isaaclab_max = ISAACLAB_LIMITS[i]
            lerobot_min, lerobot_max = LEROBOT_LIMITS[i]
            isaac_range = isaaclab_max - isaaclab_min
            lerobot_range = lerobot_max - lerobot_min
            result[:, i] = (result[:, i] - lerobot_min) / lerobot_range * isaac_range + isaaclab_min

        # Convert degrees to radians
        result = result * np.pi / 180.0
        return result

    def get_next_action(self):
        """Return the next action for replay_mode='action', or None if episode finished/unavailable."""
        if self.actions is not None and self.step_idx < len(self.actions):
            action = self.actions[self.step_idx]
            self.step_idx += 1
            return action
        return None

    def get_next_state(self):
        """Return the next state for replay_mode='state', or None if episode finished/unavailable."""
        if self.states is not None and self.step_idx < len(self.states):
            state = self.states[self.step_idx]
            self.step_idx += 1
            return state
        return None

    def get_initial_state(self):
        """Return the initial state dict loaded from HDF5."""
        return self.initial_state


# ---------------------------------------------------------------------
# Dataset loader: align parquet episodes with HDF5 episodes and filter invalid ones
# ---------------------------------------------------------------------
class HybridDatasetLoader:
    """
    Loads:
      - Parquet episode files: actions/states
      - HDF5 episode initial states

    Performs a pre-scan over HDF5 to filter out failed episodes (based on attrs and required keys).
    """

    def __init__(self, hdf5_path: str, parquet_dir: str):
        self.hdf5_path = hdf5_path
        self.parquet_dir = Path(parquet_dir)

        self.hdf5_handler = HDF5DatasetFileHandler()
        self.hdf5_handler.open(self.hdf5_path)

        # Collect parquet episodes and sort by the integer index in their filename (episode_XXXX.parquet).
        self.all_parquet_files = sorted(
            self.parquet_dir.glob("**/episode_*.parquet"),
            key=lambda p: int(re.search(r"episode_(\d+)", p.name).group(1))
            if re.search(r"episode_(\d+)", p.name)
            else 0,
        )

        # HDF5 episode names as stored in the dataset (e.g. "demo_0", "demo_1", ...).
        self.all_hdf5_names = list(self.hdf5_handler.get_episode_names())
        self.valid_indices = []

        # Pre-scan HDF5 episodes to filter out failures/missing keys.
        print("Pre-scanning HDF5 episode attributes to filter out failed data...")
        with h5py.File(self.hdf5_path, "r") as f:
            # Use the max range to iterate; assumes parquet and HDF5 correspond broadly.
            max_range = max(len(self.all_hdf5_names), len(self.all_parquet_files))

            for i in range(max_range):
                demo_name = self.all_hdf5_names[i]
                demo_path = f"data/{demo_name}"

                if demo_path in f:
                    demo_group = f[demo_path]
                    # If 'success' attr does not exist, default to True.
                    is_success = demo_group.attrs.get("success", True)

                    if is_success:
                        try:
                            # Validate required datasets exist.
                            _ = demo_group["actions"]
                            _ = demo_group["obs/joint_pos"]
                            _ = demo_group["obs/front"]
                            self.valid_indices.append(i)
                        except KeyError:
                            print(f"Pre-scan: Filtering out failed episode {demo_name}")
                    else:
                        print(f"Pre-scan: Filtering out failed episode {demo_name}")

        self.num_episodes = len(self.valid_indices)
        print(f"Pre-scan complete: {self.num_episodes} valid episodes (original total: {max_range}).")

    def load_episode(self, logical_index: int, device: str):
        """
        Load an episode:
          - Parquet is addressed directly by logical_index (episode ordering from parquet list)
          - HDF5 episode is chosen via valid_indices mapping

        Returns:
          HybridEpisodeData or None if out-of-range.
        """
        if logical_index >= len(self.all_parquet_files):
            return None

        parquet_path = self.all_parquet_files[logical_index]

        # Map logical index (parquet ordering) into an actual HDF5 index among valid ones.
        if logical_index < len(self.valid_indices):
            actual_hdf5_idx = self.valid_indices[logical_index]
        else:
            # If parquet list is longer than valid HDF5 list, fall back to last valid index.
            actual_hdf5_idx = self.valid_indices[-1] if self.valid_indices else 0

        demo_name = self.all_hdf5_names[actual_hdf5_idx]
        hdf5_ep_data = self.hdf5_handler.load_episode(demo_name, device)
        initial_state = hdf5_ep_data.get_initial_state()

        df = pd.read_parquet(parquet_path)
        return HybridEpisodeData(df, initial_state, device)

    def close(self):
        """Close underlying HDF5 handler."""
        self.hdf5_handler.close()


# ---------------------------------------------------------------------
# Rate limiter: attempt to replay at a fixed step rate while rendering
# ---------------------------------------------------------------------
class RateLimiter:
    """
    Simple real-time rate limiter:
    - sleeps to maintain a target hz
    - calls env.sim.render() during waiting to keep UI responsive
    """

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        # Advance the timeline by exactly one tick.
        self.last_time = self.last_time + self.sleep_duration

        # Catch up if we are behind.
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


# ---------------------------------------------------------------------
# Replay helpers
# ---------------------------------------------------------------------
def get_next_action_or_state(episode_data, replay_mode: str):
    """
    Depending on replay_mode:
      - "state": replay joint states (episode_data.states)
      - "action": replay actions (episode_data.actions)
    """
    if replay_mode == "state":
        next_state = episode_data.get_next_state()
        if next_state is None:
            return None
        return next_state
    else:
        return episode_data.get_next_action()


# ---------------------------------------------------------------------
# Replay manager: multi-env episode scheduling, stepping, success tracking
# ---------------------------------------------------------------------
class ReplayManager:
    """
    Runs replay in N parallel envs:
      - schedules episodes to env slots
      - steps the env with action/state vectors
      - tracks success based on termination signal
      - prints per-episode success/failure summary
    """

    def __init__(self, env, dataset_loader, config):
        self.env = env
        self.loader = dataset_loader
        self.cfg = config
        self.num_envs = env.num_envs
        self.device = env.device

        # If env provides a configured idle_action, use it; otherwise default to zeros.
        if hasattr(env.cfg, "idle_action"):
            self.idle_action = env.cfg.idle_action.repeat(self.num_envs, 1)
        else:
            self.idle_action = torch.zeros(env.action_space.shape, device=self.device)

        self.rate_limiter = RateLimiter(config.step_hz)
        self.task_type = get_task_type(config.task, config.task_type)
        self.success_count = 0
        self.total_episodes = 0

    def run(self, episode_indices: List[int]):
        """
        Replay the given list of episode indices.

        Episode lifecycle per env:
          - if env has no active episode: load next from queue, reset_to initial state
          - each step: pull next action/state and env.step()
          - if episode ends: record success based on termination flag observed during replay
        """
        episode_queue = list(episode_indices)

        # Per-env bookkeeping:
        # env_data_map: active HybridEpisodeData for each env_id
        # env_idx_map: which episode index is running in each env_id
        # env_success_map: whether that env has triggered termination (treated as success)
        env_data_map: Dict[int, Optional[HybridEpisodeData]] = {i: None for i in range(self.num_envs)}
        env_idx_map: Dict[int, Optional[int]] = {i: None for i in range(self.num_envs)}
        env_success_map: Dict[int, bool] = {i: False for i in range(self.num_envs)}
        recorded = set()

        print(f"Replaying {len(episode_indices)} episodes...")

        # Use inference_mode to avoid autograd overhead; suppress Ctrl+C traceback noise.
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running() and not simulation_app.is_exiting():
                actions = self.idle_action.clone()
                has_next = False

                for env_id in range(self.num_envs):
                    curr_data = env_data_map[env_id]
                    next_act = get_next_action_or_state(curr_data, self.cfg.replay_mode) if curr_data else None

                    if next_act is None:
                        # Episode finished: record success/failure once per episode index.
                        curr_idx = env_idx_map[env_id]
                        if curr_idx is not None and curr_idx not in recorded:
                            recorded.add(curr_idx)
                            self.total_episodes += 1
                            if env_success_map[env_id]:
                                self.success_count += 1
                                print(f"[Episode #{curr_idx}] ✓ SUCCESS | {self.success_count}/{self.total_episodes}")
                            else:
                                print(f"[Episode #{curr_idx}] ✗ FAILED | {self.success_count}/{self.total_episodes}")

                        # Load next episode from queue for this env slot.
                        next_idx = None
                        while episode_queue:
                            next_idx = episode_queue.pop(0)
                            if next_idx < len(self.loader.all_parquet_files):
                                break
                            next_idx = None

                        if next_idx is not None:
                            new_data = self.loader.load_episode(next_idx, self.device)
                            if new_data:
                                env_data_map[env_id] = new_data
                                env_idx_map[env_id] = next_idx
                                env_success_map[env_id] = False

                                # Reset only the selected env_id using provided initial state.
                                self.env.reset_to(
                                    new_data.get_initial_state(),
                                    torch.tensor([env_id], device=self.device),
                                    is_relative=True,
                                )

                                # After reset, immediately fetch the first action/state for this episode.
                                next_act = get_next_action_or_state(new_data, self.cfg.replay_mode)
                                has_next = True
                        else:
                            # No more episodes available.
                            env_data_map[env_id] = None
                            env_idx_map[env_id] = None
                            continue
                    else:
                        has_next = True

                    # Apply action/state for this env slot.
                    if next_act is not None:
                        actions[env_id] = next_act

                # If no env has any next step, exit.
                if not has_next:
                    break

                # Optionally update gripper effort limits dynamically between steps.
                if self.cfg.replay_mode == "action" and self.env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(self.env, self.task_type)

                # Step simulation with batched actions.
                _, _, reset_terminated, _, _ = self.env.step(actions)

                # Mark success if termination triggered for that env slot.
                for env_id in range(self.num_envs):
                    if reset_terminated[env_id] and env_idx_map[env_id] is not None:
                        env_success_map[env_id] = True

                # Maintain wall-clock rate and render while waiting.
                self.rate_limiter.sleep(self.env)

        # Print final success rate summary.
        success_rate = (self.success_count / max(1, self.total_episodes)) * 100
        print(f"\n{'='*50}")
        print(f"Success Rate: {self.success_count}/{self.total_episodes} ({success_rate:.1f}%)")
        print(f"{'='*50}")


# ---------------------------------------------------------------------
# Environment creation helper
# ---------------------------------------------------------------------
def create_env(args):
    """
    Create and initialize IsaacLab environment with appropriate task config.
    Also configures teleop device and disables recorders/timeouts where appropriate.
    """
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    task_type = get_task_type(args.task, args.task_type)
    env_cfg.use_teleop_device(task_type)

    # Some tasks (Direct) use manual termination, so disable timeouts/recorders.
    if "Direct" in args.task:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
        env_cfg.recorders = {}
    else:
        env_cfg.recorders = {}
        # Disable time_out termination if present.
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    return env


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def main():
    # Load datasets (parquet + hdf5) and pre-filter invalid episodes.
    dataset_loader = HybridDatasetLoader(args_cli.hdf5_file, args_cli.parquet_dir)
    print(f"Datasets loaded. Available episodes: {dataset_loader.num_episodes}")

    # If no episodes are explicitly selected, replay all parquet episodes found.
    episode_indices = args_cli.select_episodes if args_cli.select_episodes else list(range(len(dataset_loader.all_parquet_files)))
    episode_indices = [idx for idx in episode_indices if idx < len(dataset_loader.all_parquet_files)]

    if not episode_indices:
        print("No valid episodes to replay.")
        return

    print(f"Selected {len(episode_indices)} episodes for replay.")

    # Create env and run replay.
    env = create_env(args_cli)
    manager = ReplayManager(env, dataset_loader, args_cli)
    manager.run(episode_indices)

    # Cleanup resources.
    env.close()
    dataset_loader.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
