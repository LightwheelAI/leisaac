"""Script to train pick-orange RL agent with RSL-RL (PPO)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train pick-orange RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=3000, help="Number of PPO iterations.")
parser.add_argument("--log_dir", type=str, default="logs/rl/pick_orange", help="Logging directory.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video", action="store_true", default=False, help="Record evaluation videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from datetime import datetime

import gymnasium as gym
import leisaac.tasks  # noqa: F401 — registers all leisaac gym envs
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

TASK_NAME = "LeIsaac-SO101-PickOrange-RL-v0"


def main():
    from leisaac.tasks.pick_orange.agents.rsl_rl_ppo_cfg import PickOrangeRLPPORunnerCfg
    from leisaac.tasks.pick_orange.pick_orange_rl_env_cfg import PickOrangeRLEnvCfg

    env_cfg = PickOrangeRLEnvCfg()
    agent_cfg = PickOrangeRLPPORunnerCfg()

    # Override from CLI
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    # Set up logging directory
    log_dir = os.path.join(args_cli.log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir = os.path.abspath(log_dir)
    print(f"[INFO] Logging to: {log_dir}")
    env_cfg.log_dir = log_dir

    # Create environment
    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create PPO runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Dump configs
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
