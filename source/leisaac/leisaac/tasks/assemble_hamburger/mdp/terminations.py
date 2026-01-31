from __future__ import annotations

import torch
from isaaclab.assets import DeformableObject, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def burger_assembly_success(
    env: ManagerBasedRLEnv | DirectRLEnv,
    beef_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg,
    xy_threshold: float = 0.045,
    z_threshold: float = 0.03,
) -> torch.Tensor:
    """Check if the burger beef is placed on the plate.

    Success condition:
    - XY distance between beef center and plate center < xy_threshold
    - Z distance < z_threshold

    Args:
        env: The RL environment instance.
        beef_cfg: Configuration for the beef deformable object.
        plate_cfg: Configuration for the plate rigid object.
        xy_threshold: Maximum XY distance for success (default 0.045m).
        z_threshold: Maximum Z distance for success (default 0.03m).

    Returns:
        A boolean tensor indicating success for each environment.
    """
    # Access deformable object from deformable_objects dict
    beef: DeformableObject = env.scene.deformable_objects[beef_cfg.name]
    # Access rigid object from rigid_objects dict
    plate: RigidObject = env.scene.rigid_objects[plate_cfg.name]

    beef_pos = beef.data.root_pos_w  # (num_envs, 3)
    plate_pos = plate.data.root_pos_w  # (num_envs, 3)

    # XY distance
    diff_xy = beef_pos[:, :2] - plate_pos[:, :2]
    dist_xy = torch.linalg.norm(diff_xy, dim=-1)

    # Z distance
    diff_z = torch.abs(beef_pos[:, 2] - plate_pos[:, 2])

    # Success condition
    success_mask = (dist_xy < xy_threshold) & (diff_z < z_threshold)

    # Debug print
    if success_mask.any():
        print(f"[BURGER SUCCESS] xy_dist={dist_xy[0]:.4f}, z_dist={diff_z[0]:.4f}")

    return success_mask
