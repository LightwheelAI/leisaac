"""Generic script to play any leisaac RL task with a trained RSL-RL checkpoint."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a leisaac RL task with RSL-RL.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (e.g. LeIsaac-SO101-LiftCube-RL-v0).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=0, help="Episodes to run per env (0 = infinite).")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force rendering on
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib

import gymnasium as gym
import leisaac.tasks  # noqa: F401 — registers all leisaac gym envs
import torch
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
    env_cfg.recorders = None  # disable HDF5 recording during eval

    device = env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # --- Load checkpoint ---
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    policy = runner.get_inference_policy(device=device)

    # --- Episode tracking ---
    episode_count = 0
    success_count = 0

    step = 0
    obs = env.get_observations()
    while simulation_app.is_running() and (args_cli.num_episodes <= 0 or episode_count < args_cli.num_episodes):
        with torch.no_grad():
            actions = policy(obs)
        # Print actions, obs, and heights every 10 steps for env 0
        if step % 10 == 0:
            a = actions[0].cpu().numpy()
            o = obs["policy"][0].cpu().numpy()
            # obs layout: joint_pos(6) | joint_vel(6) | ee_state(7) | cube_rel_ee(3) | cube_quat(4)
            cube_rel = o[19:22]  # cube position relative to EE jaw (dx, dy, dz)

            # Raw world heights for debugging
            unwrapped = env.unwrapped
            robot = unwrapped.scene["robot"]
            cube = unwrapped.scene["cube"]
            base_index = robot.data.body_names.index("base")
            robot_base_z = robot.data.body_pos_w[0, base_index, 2].item()
            cube_z = cube.data.root_pos_w[0, 2].item()
            height_above_base = cube_z - robot_base_z

            print(
                f"[step {step:4d}] actions: {' '.join(f'{v:+.3f}' for v in a)}"
                f"  cube_rel_ee(dx,dy,dz): {' '.join(f'{v:+.3f}' for v in cube_rel)}"
                f"  cube_z={cube_z:.3f} base_z={robot_base_z:.3f} height_above_base={height_above_base:.3f}"
            )
        step += 1
        obs, _, dones, extras = env.step(actions)

        time_outs = extras.get("time_outs", torch.zeros_like(dones))
        finished = dones.bool()
        if finished.any():
            successes = finished & ~time_outs.bool()
            episode_count += int(finished.sum().item())
            success_count += int(successes.sum().item())
            success_rate = success_count / episode_count if episode_count > 0 else 0.0
            print(
                f"[Episode {episode_count}] "
                f"success={int(successes.sum().item())} "
                f"timeout={int((finished & time_outs.bool()).sum().item())} "
                f"| total success rate: {success_rate:.1%} ({success_count}/{episode_count})"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
