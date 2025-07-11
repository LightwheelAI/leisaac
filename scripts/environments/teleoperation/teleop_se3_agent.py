# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn",force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", choices=['keyboard', 'so101leader'], help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", default=False, help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")

parser.add_argument("--recalibrate", action="store_true", default=False, help="recalibrate SO101-Leader")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from leisaac.devices import Se3Keyboard, SO101Leader
from leisaac.enhance.managers import StreamingRecorderManager

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

def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    task_name = args_cli.task

    # modify configuration
    env_cfg.terminations.time_out = None
    if args_cli.record:
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
    else:
        env_cfg.recorders = None
    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    # replace the original recorder manager with the streaming recorder manager
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = 'lzf'

    # create controller
    if args_cli.teleop_device == "keyboard":
        teleop_interface = Se3Keyboard(env, sensitivity=0.25 * args_cli.sensitivity)
    # elif args_cli.teleop_device.startswith("vr"):
    #     image_size = (720, 1280)
    #     shm = shared_memory.SharedMemory(
    #         create=True,
    #         size=image_size[0] * image_size[1] * 3 * np.uint8().itemsize,
    #     )
    #     vr_device_type = {
    #         "vr-controller": VRController,
    #         "vr-hand": VRHand,
    #     }[args_cli.teleop_device.lower()]
    #     teleop_interface = vr_device_type(env,
    #                                     img_shape=image_size,
    #                                     shm_name=shm.name,)
    elif args_cli.teleop_device == "so101leader":
        teleop_interface = SO101Leader(env, recalibrate=args_cli.recalibrate)
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'vr', 'so101leader'."
        )

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    env.reset()
    teleop_interface.reset()

    current_recorded_demo_count = 0

    start_record_state = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = teleop_interface.advance()
            if actions is None or should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
                if start_record_state == True:
                    print("Stop Recording!!!")
                    start_record_state = False
            elif isinstance(actions, bool) and actions == False:
                env.render()
            # apply actions
            else:
                if start_record_state == False:
                    print("Start Recording!!!")
                    start_record_state = True
                env.step(actions)

                # print out the current demo count if it has changed
                if args_cli.record and env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                
                if args_cli.record and args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break
            if rate_limiter:
                rate_limiter.sleep(env)

    # close the simulator
    env.close()
    simulation_app.close()



if __name__ == "__main__":
    # run the main function
    main()