from dataclasses import MISSING
from typing import Any

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.devices.action_process import preprocess_device_action
from leisaac.enhance.envs import RecorderEnhanceDirectRLEnv as DirectRLEnv
from leisaac.enhance.envs import RecorderEnhanceDirectRLEnvCfg as DirectRLEnvCfg
from leisaac.enhance.envs.mdp.recorders.recorders_cfg import (
    DirectEnvActionStateRecorderManagerCfg as RecordTerm,
)

from .. import mdp
from ..bi_arm_env_cfg import BiArmEventCfg, BiArmTaskSceneCfg


@configclass
class BiArmTaskDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for the bi arm direct task environment."""

    scene: BiArmTaskSceneCfg = MISSING

    events: BiArmEventCfg = BiArmEventCfg()

    recorders: RecordTerm = RecordTerm()

    # space
    action_space = 12
    state_space = {
        "left_joint_pos": 6,
        "left_joint_vel": 6,
        "left_joint_pos_rel": 6,
        "left_joint_vel_rel": 6,
        "left_joint_pos_target": 6,
        "right_joint_pos": 6,
        "right_joint_vel": 6,
        "right_joint_pos_rel": 6,
        "right_joint_vel_rel": 6,
        "right_joint_pos_target": 6,
        "actions": action_space,
    }
    observation_space = {
        "left_joint_pos": 6,
        "right_joint_pos": 6,
        "left_joint_pos_target": 6,
        "right_joint_pos_target": 6,
        "actions": action_space,
    }

    action_scale = 1.0

    dynamic_reset_gripper_effort_limit: bool = True
    """Whether to dynamically reset the gripper effort limit."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 25.0
        self.viewer.eye = (2.5, -1.0, 1.3)
        self.viewer.lookat = (3.6, -0.4, 1.0)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

        self.scene.left_arm.init_state.pos = (3.4, -0.65, 0.89)
        self.scene.right_arm.init_state.pos = (3.8, -0.65, 0.89)

        self.cameras = []
        for cam in ["left_wrist", "right_wrist", "top"]:
            if hasattr(self.scene, cam):
                self.state_space[cam] = [getattr(self.scene, cam).height, getattr(self.scene, cam).width, 3]
                self.observation_space[cam] = [getattr(self.scene, cam).height, getattr(self.scene, cam).width, 3]
                self.cameras.append(cam)

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        # self.actions = init_action_cfg(self.actions, device=teleop_device)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)


class BiArmTaskDirectEnv(DirectRLEnv):
    """Direct RL Environment for bi arm task"""

    cfg: BiArmTaskDirectEnvCfg

    def __init__(self, cfg: BiArmTaskDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _setup_scene(self):
        """Setup the scene with more user-defined options"""
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # TODO: support more teleop_device like keyboard
        self.actions = actions.clone() * self.cfg.action_scale

    def _apply_action(self) -> None:
        left_arm_action = self.actions[:, 0:6]
        right_arm_action = self.actions[:, 6:12]
        self.scene["left_arm"].set_joint_position_target(left_arm_action)
        self.scene["right_arm"].set_joint_position_target(right_arm_action)

    def _get_observations(self) -> dict:
        obs = {
            "policy": {
                "left_joint_pos": mdp.joint_pos(self, asset_cfg=SceneEntityCfg("left_arm")),
                "left_joint_vel": mdp.joint_vel(self, asset_cfg=SceneEntityCfg("left_arm")),
                "left_joint_pos_rel": mdp.joint_pos_rel(self, asset_cfg=SceneEntityCfg("left_arm")),
                "left_joint_vel_rel": mdp.joint_vel_rel(self, asset_cfg=SceneEntityCfg("left_arm")),
                "left_joint_pos_target": mdp.joint_pos_target(self, asset_cfg=SceneEntityCfg("left_arm")),
                "right_joint_pos": mdp.joint_pos(self, asset_cfg=SceneEntityCfg("right_arm")),
                "right_joint_vel": mdp.joint_vel(self, asset_cfg=SceneEntityCfg("right_arm")),
                "right_joint_pos_rel": mdp.joint_pos_rel(self, asset_cfg=SceneEntityCfg("right_arm")),
                "right_joint_vel_rel": mdp.joint_vel_rel(self, asset_cfg=SceneEntityCfg("right_arm")),
                "right_joint_pos_target": mdp.joint_pos_target(self, asset_cfg=SceneEntityCfg("right_arm")),
                "actions": self.actions,
            }
        }
        for cam in self.cfg.cameras:
            obs["policy"][cam] = mdp.image(self, sensor_cfg=SceneEntityCfg(cam), data_type="rgb", normalize=False)
        return obs

    def _get_rewards(self) -> torch.Tensor:
        return 0.0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.manual_terminate and self.cfg.return_success_status:
            done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.never_time_out:
            time_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        if not self.cfg.manual_terminate:
            done = self._check_success()
        return done, time_out

    def _check_success(self) -> torch.Tensor:
        raise NotImplementedError(f"check_success is not implemented for {self.__class__.__name__}.")
