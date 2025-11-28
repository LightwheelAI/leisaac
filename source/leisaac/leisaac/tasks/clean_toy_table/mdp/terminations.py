import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from leisaac.utils.robot_utils import is_so101_at_rest_pose


def objs_in_box(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg_list: list[SceneEntityCfg],
    box_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
) -> torch.Tensor:
    """Determine if the objects are in the box.

    This function checks whether all success conditions for the task have been met:
    1. objects are within the target x/y range
    2. robot come back to the rest pose

    Args:
        env: The RL environment instance.
        object_cfg_list: Configuration for the object entities.
        box_cfg: Configuration for the box entity.
        x_range: Range of x positions of the object for task completion.
        y_range: Range of y positions of the object for task completion.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    box: RigidObject = env.scene[box_cfg.name]
    box_x = box.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    box_y = box.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]

    for object_cfg in object_cfg_list:
        object: RigidObject = env.scene[object_cfg.name]
        object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
        done = torch.logical_and(done, object_x < box_x + x_range[1])
        done = torch.logical_and(done, object_x > box_x + x_range[0])
        done = torch.logical_and(done, object_y < box_y + y_range[1])
        done = torch.logical_and(done, object_y > box_y + y_range[0])

    if "robot" in env.scene.keys():
        done = torch.logical_and(
            done, is_so101_at_rest_pose(env.scene["robot"].data.joint_pos, env.scene["robot"].data.joint_names)
        )
    else:
        done = torch.logical_and(
            done, is_so101_at_rest_pose(env.scene["left_arm"].data.joint_pos, env.scene["left_arm"].data.joint_names)
        )
        done = torch.logical_and(
            done, is_so101_at_rest_pose(env.scene["right_arm"].data.joint_pos, env.scene["right_arm"].data.joint_names)
        )

    return done
