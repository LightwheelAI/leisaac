"""Script to train lift-cube RL agent with RSL-RL (PPO)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train lift-cube RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=1500, help="Number of PPO iterations.")
parser.add_argument("--log_dir", type=str, default="logs/rl/lift_cube", help="Logging directory.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from datetime import datetime

import gymnasium as gym
import leisaac.tasks  # noqa: F401 — registers all leisaac gym envs
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

TASK_NAME = "LeIsaac-SO101-LiftCube-RL-v0"

# Train config for new rsl_rl API (actor/critic/algorithm top-level keys)
TRAIN_CFG = {
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
        },
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.005,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 3.0e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    # obs_groups maps actor/critic to the "policy" observation group from the env
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "num_steps_per_env": 24,
    "save_interval": 50,
    "experiment_name": "lift_cube_rl",
    "seed": 42,
}


def main():
    from leisaac.tasks.lift_cube.lift_cube_rl_env_cfg import LiftCubeRLEnvCfg

    env_cfg = LiftCubeRLEnvCfg()

    # Override from CLI
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    train_cfg = dict(TRAIN_CFG)
    train_cfg["seed"] = args_cli.seed

    device = env_cfg.sim.device

    # Set up logging directory
    log_dir = os.path.join(args_cli.log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir = os.path.abspath(log_dir)
    print(f"[INFO] Logging to: {log_dir}")
    env_cfg.log_dir = log_dir

    # Create environment
    env = gym.make(TASK_NAME, cfg=env_cfg)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create PPO runner
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)

    # Dump env config
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    # Train
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
