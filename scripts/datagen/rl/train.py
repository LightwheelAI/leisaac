"""Train a leisaac RL task with RSL-RL (PPO)."""

import argparse
import importlib
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a leisaac RL task with RSL-RL.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (e.g. LeIsaac-SO101-LiftCube-RL-v0).")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=1500, help="Number of PPO iterations.")
parser.add_argument("--log_dir", type=str, default="logs/rl", help="Base logging directory.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt) to resume training from.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def _load_from_entry_point(entry_point: str):
    module_path, attr = entry_point.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), attr)


def main():
    kwargs = gym.registry[args_cli.task].kwargs
    env_cfg = _load_from_entry_point(kwargs["env_cfg_entry_point"])()
    train_cfg = dict(_load_from_entry_point(kwargs["rsl_rl_cfg_entry_point"]))

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    train_cfg["seed"] = args_cli.seed

    task_slug = args_cli.task.lower().replace("-", "_").replace("leisaac_", "").replace("_v0", "")
    log_dir = os.path.abspath(os.path.join(args_cli.log_dir, task_slug, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    env_cfg.log_dir = log_dir
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Logging to: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=env_cfg.sim.device)
    if args_cli.checkpoint is not None:
        runner.load(args_cli.checkpoint)
        print(f"[INFO] Resumed from checkpoint: {args_cli.checkpoint}")

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
