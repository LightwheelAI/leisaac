"""Script to play lift-cube RL agent with a trained checkpoint."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play lift-cube RL agent.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force rendering on
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import leisaac.tasks  # noqa: F401
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

TASK_NAME = "LeIsaac-SO101-LiftCube-RL-v0"

TRAIN_CFG = {
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
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
        "clip_param": 0.1,
        "entropy_coef": 0.001,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 3.0e-4,
        "schedule": "fixed",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 0.5,
    },
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "num_steps_per_env": 48,
    "save_interval": 50,
    "experiment_name": "lift_cube_rl",
    "seed": 42,
}


def main():
    from leisaac.tasks.lift_cube.lift_cube_rl_env_cfg import LiftCubeRLEnvCfg

    env_cfg = LiftCubeRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    device = env_cfg.sim.device

    env = gym.make(TASK_NAME, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load checkpoint
    runner = OnPolicyRunner(env, TRAIN_CFG, log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    policy = runner.get_inference_policy(device=device)

    # Run inference loop
    obs = env.get_observations()
    step = 0
    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
