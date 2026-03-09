"""Generic script to train any leisaac RL task with RSL-RL (PPO)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a leisaac RL task with RSL-RL.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (e.g. LeIsaac-SO101-LiftCube-RL-v0).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=1500, help="Number of PPO iterations.")
parser.add_argument("--log_dir", type=str, default="logs/rl", help="Base logging directory.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
import os
from datetime import datetime

import gymnasium as gym
import leisaac.tasks  # noqa: F401 — registers all leisaac gym envs
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def _load_from_entry_point(entry_point: str):
    """Load an object from a 'module.path:attribute' entry point string."""
    module_path, attr = entry_point.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def main():
    # --- Load configs from task registry ---
    task_spec = gym.registry[args_cli.task]
    kwargs = task_spec.kwargs

    env_cfg_cls = _load_from_entry_point(kwargs["env_cfg_entry_point"])
    train_cfg = dict(_load_from_entry_point(kwargs["rsl_rl_cfg_entry_point"]))

    # --- Build env ---
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    train_cfg["seed"] = args_cli.seed

    device = env_cfg.sim.device

    # --- Set up logging ---
    task_slug = args_cli.task.lower().replace("-", "_").replace("leisaac_", "").replace("_v0", "")
    log_dir = os.path.join(args_cli.log_dir, task_slug, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir = os.path.abspath(log_dir)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Logging to: {log_dir}")
    env_cfg.log_dir = log_dir

    # --- Create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # --- Create PPO runner ---
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)

    # --- Dump env config ---
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    # --- Train ---
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
