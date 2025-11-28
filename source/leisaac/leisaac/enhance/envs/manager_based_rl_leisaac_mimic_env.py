from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


class ManagerBasedRLLeIsaacMimicEnv(ManagerBasedRLMimicEnv):
    """
    Environment for the LeIsaac task with mimic environment.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.use_teleop_device(f"mimic_{cfg.task_type}")
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_root_pos = self.scene["robot"].data.root_pos_w
        self.robot_root_quat = self.scene["robot"].data.root_quat_w
        self.task_type = cfg.task_type

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        # robot coordinate
        eef_state = self.obs_buf["policy"]["ee_frame_state"][env_ids]
        eef_pos = eef_state[:, :3]
        eef_quat = eef_state[:, 3:7]
        # quat: (w, x, y, z)
        eef_pose = PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))
        return eef_pose

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_eef_pos, target_eef_rot = PoseUtils.unmake_pose(target_eef_pose)
        target_eef_quat = PoseUtils.quat_from_matrix(target_eef_rot)

        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([target_eef_pos, target_eef_quat], dim=0)
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise

        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        target_eef_pos = action[:, :3]
        target_eef_quat = action[:, 3:7]
        target_eef_rot = PoseUtils.matrix_from_quat(target_eef_quat)

        target_eef_pose = PoseUtils.make_pose(target_eef_pos, target_eef_rot).clone()

        return {eef_name: target_eef_pose}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        return {eef_name: actions[:, -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        rigid_object_state = self.scene.get_state(is_relative=False)["rigid_object"]
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_state.items():
            obj_pos_w, obj_quat_w = obj_state["root_pose"][env_ids, :3], obj_state["root_pose"][env_ids, 3:7]
            obj_pos_robot, obj_quat_robot = PoseUtils.subtract_frame_transforms(
                self.robot_root_pos[env_ids], self.robot_root_quat[env_ids], obj_pos_w, obj_quat_w
            )
            object_pose_matrix[obj_name] = PoseUtils.make_pose(
                obj_pos_robot, PoseUtils.matrix_from_quat(obj_quat_robot)
            )

        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        for term_name, term_signal in subtask_terms.items():
            signals[term_name] = term_signal[env_ids]

        return signals

    def step(self, action: torch.Tensor):
        if self.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(self, self.task_type)
        return super().step(action)
