import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedEnv
from leisaac.utils.robot_utils import is_so101_at_rest_pose


def sausage_cut(
    env: ManagerBasedEnv | DirectRLEnv,
    min_sausage_count: int = 2,
    check_rest_pose: bool = True,
) -> torch.Tensor:
    """Determine if the sausage cutting task is completed successfully.

    This function evaluates the success conditions for the sausage cutting task:
    1. Sausage has been cut into at least `min_sausage_count` pieces
    2. (Optional) Robot arms return to the rest pose

    Args:
        env: The RL environment instance.
        min_sausage_count: Minimum number of sausage pieces required for success (default: 2).
        check_rest_pose: Whether to check if robots are at rest pose (default: True).

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Check if sausage has been cut (access from environment's custom attribute)
    if hasattr(env, "sausage_count"):
        sausage_cut_done = env.sausage_count >= min_sausage_count
        done = torch.logical_or(done, sausage_cut_done)

    # Optionally check if robots are at rest pose
    if check_rest_pose and hasattr(env.scene, "articulations"):
        left_arm = env.scene.articulations.get("left_arm") or env.scene.get("left_arm")
        right_arm = env.scene.articulations.get("right_arm") or env.scene.get("right_arm")

        if left_arm is not None and right_arm is not None:
            is_rest = torch.logical_and(
                is_so101_at_rest_pose(left_arm.data.joint_pos, left_arm.data.joint_names),
                is_so101_at_rest_pose(right_arm.data.joint_pos, right_arm.data.joint_names),
            )
            done = torch.logical_and(done, is_rest)

    return done
