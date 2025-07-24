"""Script to run a leisaac replay with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac replay for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to load recorded demos.")
parser.add_argument("--episode_index", type=int, default=0, help="Replay the episode with the given index.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time
import torch
import h5py
import numpy as np
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

import leisaac


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def parse_dataset(dataset_file: str, episode_index: int, task_type: str, device: str):
    with h5py.File(dataset_file, 'r') as f:
        demo_names = list(f['data'].keys())
        demo_name = demo_names[episode_index]
        demo_group = f['data'][demo_name]

        seed = int(demo_group.attrs.get("seed", -1))

        if task_type == "bi-arm":
            left_joint_pos = torch.tensor(np.array(demo_group['obs/left_joint_pos']), device=device)
            right_joint_pos = torch.tensor(np.array(demo_group['obs/right_joint_pos']), device=device)
            states = torch.cat([left_joint_pos, right_joint_pos], dim=1)
        else:
            states = torch.tensor(np.array(demo_group['obs/joint_pos']), device=device)

        init_object_state = {}
        for key in list(demo_group['initial_state/rigid_object'].keys()):
            root_state = torch.tensor(demo_group[f'initial_state/rigid_object/{key}/root_pose'], device=device)
            root_velocity = torch.tensor(demo_group[f'initial_state/rigid_object/{key}/root_velocity'], device=device)
            init_object_state[key] = torch.cat([root_state, root_velocity], dim=1)

        return states, init_object_state, seed


def reset_object_state(env: ManagerBasedRLEnv, init_object_state: dict):
    for key, root_state in init_object_state.items():
        env.scene[key].write_root_state_to_sim(root_state)


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    task_type = "bi-so101leader" if "BiArm" in args_cli.task else "so101leader"
    env_cfg.use_teleop_device(task_type)

    # modify configuration
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    rate_limiter = RateLimiter(args_cli.step_hz)

    states, init_object_state, seed = parse_dataset(args_cli.dataset_file, args_cli.episode_index, task_type, args_cli.device)

    # reset environment
    env.seed(seed)
    env.reset()
    reset_object_state(env, init_object_state)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            for state in states:
                env.step(state.unsqueeze(0))
                if rate_limiter:
                    rate_limiter.sleep(env)
            env.reset()
            reset_object_state(env, init_object_state)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
