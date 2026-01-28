# controllers/pose_expert.py
"""
Pose-based expert controller.

Outputs actions with layout:
[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]
"""
import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_inv, quat_apply

def apply_triangle_offset(pos_tensor: torch.Tensor, flag: int, radius: float = 0.1) -> torch.Tensor:
    """
    Add an equilateral-triangle offset to the x/y of pos_tensor based on flag (1..3).
    Args:
        pos_tensor: (num_envs, 3) tensor
        flag: which orange (1..3)
        radius: radius in meters
    """
    import math
    idx = (flag - 1) % 3
    angle = idx * (2 * math.pi / 3)
    offset_x = radius * math.cos(angle)
    offset_y = radius * math.sin(angle)
    pos_tensor = pos_tensor.clone()
    pos_tensor[:, 0] += offset_x
    pos_tensor[:, 1] += offset_y
    return pos_tensor

def get_expert_action_pose_based(env, step_count: int, target: str, flag: int) -> torch.Tensor:
    """
    Simple scripted pose-based controller.

    Args:
        env: environment instance with scene data
        step_count: int step counter
        target: name of target object in scene (e.g., "Orange001")
        flag: integer flag (which orange) used for offsets

    Returns:
        actions: (num_envs, action_dim) torch.Tensor on same device as env
    """
    device = env.device
    num_envs = env.num_envs

    orange_pos_w = env.scene[target].data.root_pos_w.clone()
    plate_pos_w = env.scene["Plate"].data.root_pos_w.clone()
    robot_base_pos_w = env.scene["robot"].data.root_pos_w.clone()
    robot_base_quat_w = env.scene["robot"].data.root_quat_w.clone()

    target_pos_w = orange_pos_w.clone()

    # Fixed orientation (world)
    pitch = 0.0
    target_quat_w = quat_from_euler_xyz(
        torch.tensor(pitch, device=device),
        torch.tensor(0.0, device=device),
        torch.tensor(0.0, device=device),
    ).repeat(num_envs, 1)

    target_quat = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)

    # schedule and offsets
    GRIPPER = 0.1
    gripper_cmd = torch.full((num_envs, 1), 1.0, device=device)

    if step_count < 120:
        gripper_cmd[:] = 1.0
        target_pos_w[:, 2] += 0.1 + GRIPPER
    elif step_count < 150:
        gripper_cmd[:] = 1.0
        target_pos_w[:, 2] += GRIPPER
    elif step_count < 180:
        gripper_cmd[:] = -1.0
        target_pos_w[:, 2] += GRIPPER
    elif step_count < 220:
        gripper_cmd[:] = -1.0
        target_pos_w[:, 2] += 0.25
    elif step_count < 320:
        gripper_cmd[:] = -1.0
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.25
    elif step_count < 350:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += GRIPPER + 0.1
        target_pos_w = apply_triangle_offset(target_pos_w, flag)
        gripper_cmd[:] = -1.0
    elif step_count < 380:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += GRIPPER + 0.1
        target_pos_w = apply_triangle_offset(target_pos_w, flag)
        gripper_cmd[:] = 1.0
    elif step_count < 420:
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.2
        target_pos_w = apply_triangle_offset(target_pos_w, flag)
        gripper_cmd[:] = 1.0
    else:
        gripper_cmd[:] = 1.0

    # small x offset toward robot
    target_pos_w[:, 0] -= 0.03

    diff_w = target_pos_w - robot_base_pos_w
    target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)

    actions = torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)
    return actions

"""
Heuristics for detecting grasp phase.
"""
from typing import Tuple, Dict, Any
import torch

def is_grasp_phase_auto(env_local, orange_name: str, prev_gripper_t: torch.Tensor,
                        dist_thresh: float = 0.06, rel_vel_thresh: float = 0.08,
                        gripper_close_threshold: float = 0.7) -> Tuple[bool, Dict[str, Any]]:
    """
    Return (is_grasp: bool, info: dict).

    Info contains metrics useful for debugging.
    """
    info = {}
    try:
        orange = env_local.scene[orange_name]
        orange_pos = None
        device = None

        if hasattr(orange.data, "root_pos_w"):
            orange_pos = orange.data.root_pos_w  # (num_envs,3)
            device = orange_pos.device

        if device is None:
            device = getattr(env_local, "device", torch.device("cpu"))

        robot_local = env_local.scene["robot"]
        try:
            ee_idx = robot_local.data.body_names.index("gripper")
        except Exception:
            ee_idx = 0

        ee_pos = robot_local.data.body_state_w[:, ee_idx, :3]
        if ee_pos is not None:
            device = ee_pos.device

        if orange_pos is None:
            return False, {"error": "orange pos not available"}

        horiz_dist = torch.norm((orange_pos - ee_pos)[:, :2], dim=-1)
        full_dist = torch.norm((orange_pos - ee_pos), dim=-1)
        info['horiz_dist'] = horiz_dist.detach().cpu().numpy().tolist()
        info['full_dist'] = full_dist.detach().cpu().numpy().tolist()

        obj_speed = torch.zeros((orange_pos.shape[0],), device=device)
        ee_speed = torch.zeros((orange_pos.shape[0],), device=device)
        if hasattr(orange.data, "root_linvel_w"):
            try:
                obj_v = orange.data.root_linvel_w
                obj_speed = torch.norm(obj_v, dim=-1).to(device)
            except Exception:
                pass
        if hasattr(robot_local.data, "body_linvel_w"):
            try:
                v_ee = robot_local.data.body_linvel_w[:, ee_idx, :]
                ee_speed = torch.norm(v_ee, dim=-1).to(device)
            except Exception:
                pass

        rel_speed = torch.minimum(obj_speed, ee_speed)
        info['obj_speed'] = obj_speed.detach().cpu().numpy().tolist() if obj_speed.numel() else None
        info['ee_speed'] = ee_speed.detach().cpu().numpy().tolist() if ee_speed.numel() else None
        info['rel_speed'] = rel_speed.detach().cpu().numpy().tolist()

        # ensure prev_gripper_t on same device and as python list for mask
        try:
            if not isinstance(prev_gripper_t, torch.Tensor):
                prev_gripper_t = torch.as_tensor(prev_gripper_t, device=device)
            else:
                prev_gripper_t = prev_gripper_t.to(device)
            gripper_target = prev_gripper_t.view(-1).detach().cpu().numpy().tolist()
        except Exception:
            gripper_target = None
        info['gripper_target'] = gripper_target

        near_mask = (horiz_dist <= torch.as_tensor(dist_thresh, device=device))
        slow_mask = (rel_speed <= torch.as_tensor(rel_vel_thresh, device=device))

        if gripper_target is not None:
            try:
                gripper_mask = torch.tensor([gt < gripper_close_threshold for gt in gripper_target], dtype=torch.bool, device=device)
            except Exception:
                gripper_mask = torch.zeros_like(near_mask, dtype=torch.bool, device=device)
        else:
            gripper_mask = torch.zeros_like(near_mask, dtype=torch.bool, device=device)

        is_grasp_tensor = near_mask & slow_mask & gripper_mask
        is_grasp_list = is_grasp_tensor.detach().cpu().numpy().tolist()
        is_grasp_any = any(is_grasp_list)
        return is_grasp_any, info

    except Exception as e:
        return False, {"error": str(e)}
