from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the Elfin S30 arm + Robotiq 2F-140 gripper."""

S30_ASSET_PATH = str(Path(ASSETS_ROOT) / "robots" / "elfin_s30" / "S30.usd")

S30_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=S30_ASSET_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.2, -1.5, 0.89),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": -1.5708,   # -90 deg
            "joint3": 1.5708,    # +90 deg
            "joint4": 0.0,
            "joint5": -1.5708,   # -90 deg
            "joint6": 0.0,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "elfin-arm": ImplicitActuatorCfg(
            joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            effort_limit_sim=350.0,
            velocity_limit_sim=1.57,
            stiffness=349.0,
            damping=34.9,
        ),
        "robotiq-finger": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=0.5,
            stiffness=800.0,
            damping=40.0,
        ),
        # ActionGraph drove ONLY right_outer_knuckle_joint at the same 0-45° as finger_joint.
        # Ratio = +1 (same direction): both joints share identical position targets.
        # All other 4-bar joints (inner_knuckle, outer_finger, inner_finger) are
        # handled by PhysX gear/mimic constraints embedded in physics_edit.usd.
        # inner_finger_pad_joint is NVIDIA-specific (absent from standard URDF mimic list),
        # so it has no PhysX constraint and must be tracked here with ratio = -1.
        "robotiq-mimic": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_outer_knuckle_joint",
                "left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=0.5,
            stiffness=800.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Joint limits as stored in USD (degrees).
# Derived from URDF limits converted to degrees:
#   joint1: ±6.28 rad  → ±360°
#   joint2: -3.3158 ~ 0.1745 rad → -190° ~ 10°
#   joint3: ±2.9319 rad → ±168°
#   joint4/5/6: ±6.28 rad → ±360°
#   finger_joint: 0 ~ 45° (Robotiq 2F-140 opening angle)
S30_USD_JOINT_LIMITS = {
    "joint1":       (-360.0, 360.0),
    "joint2":       (-190.0,  10.0),
    "joint3":       (-168.0, 168.0),
    "joint4":       (-360.0, 360.0),
    "joint5":       (-360.0, 360.0),
    "joint6":       (-360.0, 360.0),
    "finger_joint": (   0.0,  45.0),
}

# Motor limits — the angle range (degrees) accepted by the TCP/IP SDK.
# For S30 the SDK takes joint angles directly in degrees, so limits equal USD limits.
S30_MOTOR_LIMITS = {
    "joint1":       (-360.0, 360.0),
    "joint2":       (-190.0,  10.0),
    "joint3":       (-168.0, 168.0),
    "joint4":       (-360.0, 360.0),
    "joint5":       (-360.0, 360.0),
    "joint6":       (-360.0, 360.0),
    "finger_joint": (   0.0,  45.0),
}

# Rest pose range (degrees) used to detect whether the arm has returned to home.
# Centred on the init_state joint positions converted to degrees.
S30_REST_POSE_RANGE = {
    "joint1":       (  0.0 - 30.0,   0.0 + 30.0),
    "joint2":       (-90.0 - 30.0, -90.0 + 30.0),
    "joint3":       ( 90.0 - 30.0,  90.0 + 30.0),
    "joint4":       (  0.0 - 30.0,   0.0 + 30.0),
    "joint5":       (-90.0 - 30.0, -90.0 + 30.0),
    "joint6":       (  0.0 - 30.0,   0.0 + 30.0),
    "finger_joint": (  0.0,          15.0),          # nearly open counts as rest
}
