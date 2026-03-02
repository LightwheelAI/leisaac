"""PPO training script for LeIsaac-SO101-PickOrange-RL-v0 using rsl_rl.

Usage:
    python scripts/rl/train_pick_orange.py --num_envs 64 --num_iters 1000 \\
        --headless --log_dir ./logs/pick_orange_rl

Launch Isaac Sim before running (or pass --headless).
"""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train pick-orange with rsl_rl PPO.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_iters", type=int, default=1000)
parser.add_argument("--log_dir", type=str, default="./logs/pick_orange_rl")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import leisaac.tasks  # noqa: F401 — registers all leisaac gym envs
from leisaac.rl import IsaaclabRslRlVecEnvWrapper

# ------------------------------------------------------------------
# PPO training configuration
# ------------------------------------------------------------------
TRAIN_CFG = {
    "num_steps_per_env": 24,
    "save_interval": 50,
    "obs_groups": {
        "policy": ["policy"],
        "critic": ["policy"],
    },
    "empirical_normalization": True,
    "policy": {
        "class_name": "ActorCritic",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "init_noise_std": 1.0,
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipping": True,
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 3e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
}


def main():
    os.makedirs(args_cli.log_dir, exist_ok=True)

    env_cfg = parse_env_cfg(
        "LeIsaac-SO101-PickOrange-RL-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = int(time.time())

    env = gym.make("LeIsaac-SO101-PickOrange-RL-v0", cfg=env_cfg).unwrapped

    wrapped_env = IsaaclabRslRlVecEnvWrapper(env)

    print(f"[INFO] num_envs={wrapped_env.num_envs}  "
          f"num_actions={wrapped_env.num_actions}  "
          f"max_episode_length={wrapped_env.max_episode_length}")
    obs = wrapped_env.get_observations()
    print(f"[INFO] obs['policy'] shape: {obs['policy'].shape}")

    runner = OnPolicyRunner(
        wrapped_env,
        TRAIN_CFG,
        log_dir=args_cli.log_dir,
        device=args_cli.device,
    )
    runner.learn(num_learning_iterations=args_cli.num_iters, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
