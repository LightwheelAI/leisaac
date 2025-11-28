import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from leisaac.enhance.assets import ClothObject
from leisaac.utils.robot_utils import is_so101_at_rest_pose


def cloth_folded(
    env: ManagerBasedEnv | DirectRLEnv,
    cloth_cfg: SceneEntityCfg,
    cloth_keypoints_index: list[int],
    distance_threshold: float = 0.10,
) -> torch.Tensor:
    """Determine if the cloth folding task is completed successfully.

    This function evaluates the success conditions for the cloth folding task:
    1. Cloth is properly folded (evaluated using distance between key points)
    2. Robot returns to the rest pose

    Args:
        env: The RL environment instance.
        cloth_cfg: Configuration for the cloth entity.
        cloth_keypoints_index: Indices of cloth keypoints. We use the first 6 keypoints
            corresponding to: left sleeve, left shoulder, left hem, right sleeve, right shoulder, and right hem.
        distance_threshold: Threshold for the distance among the cloth keypoints.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    is_rest = torch.logical_and(
        is_so101_at_rest_pose(env.scene["left_arm"].data.joint_pos, env.scene["left_arm"].data.joint_names),
        is_so101_at_rest_pose(env.scene["right_arm"].data.joint_pos, env.scene["right_arm"].data.joint_names),
    )
    done = torch.logical_and(done, is_rest)

    cloth: ClothObject = env.scene.particle_objects[cloth_cfg.name]
    cloth_keypoints_pos = cloth.point_positions[:, cloth_keypoints_index[:6]]
    done = torch.logical_and(
        done, torch.norm(cloth_keypoints_pos[:, 0] - cloth_keypoints_pos[:, 4]) < distance_threshold
    )  # left sleeve -> right shoulder
    done = torch.logical_and(
        done, torch.norm(cloth_keypoints_pos[:, 3] - cloth_keypoints_pos[:, 1]) < distance_threshold
    )  # right sleeve -> left shoulder
    done = torch.logical_and(
        done, torch.norm(cloth_keypoints_pos[:, 2] - cloth_keypoints_pos[:, 1]) < distance_threshold
    )  # left hem -> left shoulder
    done = torch.logical_and(
        done, torch.norm(cloth_keypoints_pos[:, 5] - cloth_keypoints_pos[:, 4]) < distance_threshold
    )  # right hem -> right shoulder

    return done
