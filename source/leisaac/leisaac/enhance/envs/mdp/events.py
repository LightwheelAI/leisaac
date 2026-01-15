from typing import Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera


def randomize_camera_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation uniformly within the given ranges.

    * It samples the camera position and rotation from the given ranges and adds them to the
      default camera position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    asset: Camera = env.scene[asset_cfg.name]

    ori_pos_w = asset.data.pos_w[env_ids]
    if convention == "ros":
        ori_quat_w = asset.data.quat_w_ros[env_ids]
    elif convention == "opengl":
        ori_quat_w = asset.data.quat_w_opengl[env_ids]
    elif convention == "world":
        ori_quat_w = asset.data.quat_w_world[env_ids]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # camera usually spawn with robot, so no need to add env_origins
    positions = ori_pos_w[:, 0:3] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)


def randomize_particle_object_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
):
    """Reset the particle object to a random position and rotation uniformly within the given ranges.

    * It samples the particle object position and rotation from the given ranges and adds them to the
      default particle object position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    particle_object = env.scene.particle_objects[asset_cfg.name]
    ori_world_pos, ori_world_quat = particle_object.get_world_poses()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    positions = ori_world_pos + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_world_quat, orientations_delta)

    particle_object.set_world_poses(positions, orientations)


def disable_rigid_body_gravity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Disable gravity for specific bodies in an articulation.

    This function disables gravity for bodies specified in the asset_cfg.body_names.
    It uses modify_rigid_body_properties to set disable_gravity=True for the specified bodies.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the change to.
        asset_cfg: Configuration specifying the asset and body names to disable gravity for.
                   Use body_names to specify which bodies to disable gravity (e.g., ".*arm.*" or ["shoulder", "elbow"]).
    """
    # Get the asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve body indices from body_names (already resolved by SceneEntityCfg)
    if asset_cfg.body_ids == slice(None):
        body_ids = list(range(asset.num_bodies))
    else:
        body_ids = asset_cfg.body_ids if isinstance(asset_cfg.body_ids, list) else [asset_cfg.body_ids]

    # Get link paths from the first environment (they follow the same pattern for all environments)
    link_paths = asset.root_physx_view.link_paths[0]

    # Disable gravity for each specified body
    for body_id in body_ids:
        if body_id >= len(link_paths):
            continue

        # Get the link path from first environment
        first_env_link_path = link_paths[body_id]

        # Convert to regex expression by replacing env_0 with env_.*
        link_path_expr = first_env_link_path.replace("/env_0/", "/env_.*/")

        # Resolve all matching prim paths and apply
        prim_paths = sim_utils.find_matching_prim_paths(link_path_expr)
        for prim_path in prim_paths:
            sim_utils.modify_rigid_body_properties(
                prim_path,
                sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            )


def reset_mixed_objects_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    rigid_asset_cfg: list[SceneEntityCfg],
    deformable_asset_cfg: list[SceneEntityCfg],
):
    """Reset mixed rigid and deformable objects together with the same randomization offset.

    This allows rigid and deformable objects to be randomized as a group,
    sharing the same random offset across all objects.
    """
    # Sample common noise (pose)
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples_pose = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    # Sample common noise (velocity)
    range_list_vel = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges_vel = torch.tensor(range_list_vel, device=env.device)
    rand_samples_vel = math_utils.sample_uniform(
        ranges_vel[:, 0], ranges_vel[:, 1], (len(env_ids), 6), device=env.device
    )

    # 1. Handle Rigid Objects
    for cfg in rigid_asset_cfg:
        asset = env.scene[cfg.name]
        root_states = asset.data.default_root_state[env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples_pose[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples_pose[:, 3], rand_samples_pose[:, 4], rand_samples_pose[:, 5]
        )
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        velocities = root_states[:, 7:13] + rand_samples_vel
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    # 2. Handle Deformable Objects
    for cfg in deformable_asset_cfg:
        asset = env.scene[cfg.name]
        nodal_state = asset.data.default_nodal_state_w[env_ids].clone()

        nodal_state[..., :3] += rand_samples_pose[:, 0:3].unsqueeze(1)
        nodal_state[..., 3:] += rand_samples_vel[:, 0:3].unsqueeze(1)
        asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
