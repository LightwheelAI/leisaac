import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def object_in_container(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg,
    container_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
    height_threshold: float = 0.05,
) -> torch.Tensor:
    """Determine if the object is in the container.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range and below the container height threshold

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity.
        container_cfg: Configuration for the container entity.
        x_range: Range of x positions of the object for task completion.
        y_range: Range of y positions of the object for task completion.
        height_threshold: Threshold for the object height above the container.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    container: RigidObject = env.scene[container_cfg.name]
    container_x = container.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    container_y = container.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]

    object: RigidObject = env.scene[object_cfg.name]
    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_z = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    done = torch.logical_and(done, object_x < container_x + x_range[1])
    done = torch.logical_and(done, object_x > container_x + x_range[0])
    done = torch.logical_and(done, object_y < container_y + y_range[1])
    done = torch.logical_and(done, object_y > container_y + y_range[0])
    done = torch.logical_and(done, object_z < height_threshold)

    return done
