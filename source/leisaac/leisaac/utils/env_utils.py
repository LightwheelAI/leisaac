import torch
import math


def dynamic_reset_gripper_effort_limit_sim(env, teleop_device):
    need_to_set = []
    if teleop_device == "bi-so101leader":
        need_to_set = [env.scene.articulations['left_arm'], env.scene.articulations['right_arm']]
    elif teleop_device in ["so101leader", "keyboard"]:
        need_to_set = [env.scene['robot']]
    for arm in need_to_set:
        write_gripper_effort_limit_sim(env, arm)
    return


def write_gripper_effort_limit_sim(env, env_arm):
    gripper_pos = env_arm.data.body_link_pos_w[0][-1]
    minm_distance = 1e10
    target_name = None
    for name, obj in env.scene._rigid_objects.items():
        pos = obj.data.body_link_pos_w[0][0]
        distance = torch.sqrt(torch.sum((gripper_pos - pos) ** 2))
        if distance < minm_distance:
            minm_distance = distance
            target_name = name
    if target_name is not None:
        target_mass = env.scene._rigid_objects[target_name].data.default_mass
        target_effort_limit_sim = target_mass / 0.15
        current_effort_limit_sim = env_arm._data.joint_effort_limits[0][-1].item()
        if math.fabs(target_effort_limit_sim - current_effort_limit_sim) > 0.1:
            env_arm.write_joint_effort_limit_to_sim(limits=target_effort_limit_sim, joint_ids=[5])
    return
