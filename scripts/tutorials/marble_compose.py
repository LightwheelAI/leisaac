"""
Scene Composition Tool

Composes USD scenes by transforming background to align with robot's original position,
then placing task objects at their original positions.

Task types: toys, orange, cloth, cube
"""

import argparse
from pathlib import Path

import numpy as np

# Configuration

TASK_TO_ENV = {
    "toys": {
        "single": "LeIsaac-SO101-CleanToyTable-v0",
        "dual": "LeIsaac-SO101-CleanToyTable-BiArm-v0",
    },
    "orange": {
        "single": "LeIsaac-SO101-PickOrange-v0",
    },
    "cloth": {
        "dual": "LeIsaac-SO101-FoldCloth-BiArm-v0",
    },
    "cube": {
        "single": "LeIsaac-SO101-PickCube-v0",
    },
}

TASK_CONFIG = {
    "toys": {
        "objects": [
            "Kit1_Box",
            "Kit1_Bridge",
            "Kit1_Character_E",
            "Kit1_Character_G",
            "Kit1_Character_H",
            "Kit1_Character_I",
            "Kit1_Character_L",
            "Kit1_Character_Lcap",
            "Kit1_Character_T",
            "Kit1_Character_W",
            "Kit1_Cross",
            "Kit1_Cube3x3",
            "Kit1_Cube6x6",
            "Kit1_Cuboid6x3",
            "Kit1_Cylinder",
            "Kit1_Icosphere",
            "Kit1_Sphere",
            "Kit1_Torus",
            "Kit1_Triangle",
            "Kit1_Character_H_01",
            "Kit1_Character_E_01",
            "KidRoom_Table01",
        ],
        "source_scene": "scenes/lightwheel_toyroom/scene.usd",
        "assets_subpath": "scenes/lightwheel_toyroom/Assets",
        "table_name": "KidRoom_Table01",
    },
    "orange": {
        "objects": ["Orange001", "Orange002", "Orange003", "Plate"],
        "source_scene": "scenes/kitchen_with_orange/scene.usd",
        "assets_subpath": "scenes/kitchen_with_orange/objects",
    },
    "cloth": {
        "objects": ["Table038_01", "cloth"],
        "source_scene": "scenes/lightwheel_bedroom/scene.usd",
        "assets_subpath": "scenes/lightwheel_bedroom",
        "table_name": "Table038_01",
    },
    "cube": {
        "objects": ["cube"],
        "source_scene": "scenes/table_with_cube/scene.usd",
        "assets_subpath": "scenes/table_with_cube/cube",
    },
}


# Transform Utilities


def quat_to_matrix(q):
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix"""
    w, x, y, z = np.array(q) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """3x3 rotation matrix -> quaternion [w,x,y,z]"""
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        return [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        return [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        return [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        return [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]


def pose_to_matrix(pos, quat):
    """Pose -> 4x4 SE(3) matrix"""
    M = np.eye(4)
    M[:3, :3] = quat_to_matrix(quat)
    M[:3, 3] = pos
    return M


def matrix_to_pose(M):
    """4x4 SE(3) matrix -> (pos, quat)"""
    return M[:3, 3].tolist(), matrix_to_quat(M[:3, :3])


def compute_scene_transform(orig_pos, orig_quat, target_pos, target_quat):
    """Compute inverse transform: T_scene = M_orig @ M_target^(-1)"""
    M_orig = pose_to_matrix(orig_pos, orig_quat)
    M_target = pose_to_matrix(target_pos, target_quat)
    T_inv = M_orig @ np.linalg.inv(M_target)
    return matrix_to_pose(T_inv)


# USD Utilities


def get_object_usd_path(task_type: str, obj_name: str, assets_base: str) -> str:
    """Resolve USD file path for an object"""
    config = TASK_CONFIG[task_type]
    table_name = config.get("table_name")

    if task_type == "toys":
        if obj_name == table_name:
            return f"{assets_base}/{obj_name}/{obj_name}.usd"
        name = obj_name[:-3] if obj_name.endswith("_01") else obj_name
        return f"{assets_base}/Kit1/{name}.usd"

    if task_type == "orange":
        return f"{assets_base}/{obj_name}/{obj_name}.usd"

    if task_type == "cloth":
        if obj_name == table_name:
            folder = obj_name[:-3] if obj_name.endswith("_01") else obj_name
            return f"{assets_base}/LW_Loft/Loft/{folder}/{folder}.usd"
        return f"{assets_base}/cloth/cloth.usd"

    if task_type == "cube":
        return f"{assets_base}/cube.usd"

    raise ValueError(f"Unknown object: {obj_name}")


def read_layout_from_usd(usd_path: str, object_names: list[str]) -> dict[str, dict]:
    """Read object poses from USD (first-level children of root prim)"""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Cannot open: {usd_path}")

    root = stage.GetDefaultPrim() or stage.GetPrimAtPath("/")
    object_set = set(object_names)
    layout = {}

    for child in root.GetChildren():
        name = child.GetName()
        if name not in object_set:
            continue

        xform = UsdGeom.XformCache().GetLocalToWorldTransform(child)
        t = xform.ExtractTranslation()
        r = xform.ExtractRotationQuat()

        layout[name] = {"pos": [t[0], t[1], t[2]], "rot": [r.GetReal(), *r.GetImaginary()]}

    # Report missing
    for name in object_names:
        if name not in layout:
            print(f"[WARN] '{name}' not found in USD")

    return layout


def load_robot_pose(task_type: str, use_dual_arm: bool = False) -> dict[str, list[float]]:
    """Load robot init pose from EnvCfg (requires Isaac Sim running)

    Args:
        task_type: Task type (toys, orange, cloth, cube)
        use_dual_arm: If True, use dual-arm configuration and read left_arm pose

    Returns:
        Dictionary with 'pos' and 'quat' keys
    """
    from isaaclab_tasks.utils import parse_env_cfg

    # Select environment based on dual-arm flag
    task_env = TASK_TO_ENV.get(task_type)
    if not task_env:
        raise ValueError(f"Unknown task type: {task_type}")

    if use_dual_arm:
        env_id = task_env.get("dual")
        if not env_id:
            print(f"[WARN] Dual-arm not supported for '{task_type}', using single-arm")
            env_id = task_env["single"]
            use_dual_arm = False
        else:
            print(f"[INFO] Using dual-arm config: {env_id}")
    else:
        env_id = task_env["single"]

    env_cfg = parse_env_cfg(env_id, device="cpu", num_envs=1)

    # For dual-arm, read left_arm as reference
    if use_dual_arm:
        init = env_cfg.scene.left_arm.init_state
        print("[INFO] Using left_arm as reference")
    else:
        init = env_cfg.scene.robot.init_state

    return {"pos": list(init.pos), "quat": list(init.rot)}


# Main Functions


def compose_scene(
    task_type: str,
    background_usd: str,
    output_usd: str,
    assets_base: str,
    target_pos: list[float],
    target_quat: list[float],
    include_table: bool = False,
    use_dual_arm: bool = False,
) -> str:
    """
    Compose USD scene:
    1. Transform background to align with robot's original position
    2. Place objects at their original positions

    Args:
        include_table: Include table in output (default: False)
        use_dual_arm: Use dual-arm configuration (toys/cloth only, uses left_arm as reference)
    """
    from pxr import Gf, Usd, UsdGeom

    config = TASK_CONFIG[task_type]
    table_name = config.get("table_name")

    # Warn if table not supported
    if include_table and not table_name:
        print(f"[WARN] Table not supported for '{task_type}'")
        include_table = False

    # Build source USD path and read layout
    source_usd = f"{assets_base}/{config['source_scene']}"
    print(f"[INFO] Reading layout from: {source_usd}")
    layout = read_layout_from_usd(source_usd, config["objects"])

    # Compute scene transform based on reference point
    if include_table and table_name:
        # Use table as reference point
        table_pose = layout.get(table_name)
        if not table_pose:
            raise RuntimeError(f"Table '{table_name}' not found in source USD")
        orig_pos, orig_quat = table_pose["pos"], table_pose["rot"]
        print(f"[INFO] Using table '{table_name}' as reference: pos={orig_pos}")
    else:
        # Use robot as reference point (dual-arm uses left_arm)
        robot = load_robot_pose(task_type, use_dual_arm=use_dual_arm)
        orig_pos, orig_quat = robot["pos"], robot["quat"]
        print(f"[INFO] Using robot as reference: pos={orig_pos}")

    scene_pos, scene_quat = compute_scene_transform(orig_pos, orig_quat, target_pos, target_quat)

    # Filter out table if not included
    if not include_table and table_name:
        layout = {k: v for k, v in layout.items() if k != table_name}

    # Create output stage
    stage = Usd.Stage.CreateNew(output_usd)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Add background with transform
    bg_prim = stage.DefinePrim("/World/Scene")
    bg_prim.GetReferences().AddReference(background_usd)
    bg_xform = UsdGeom.Xformable(bg_prim)
    bg_xform.ClearXformOpOrder()
    bg_xform.AddTranslateOp().Set(Gf.Vec3d(*scene_pos))
    bg_xform.AddOrientOp().Set(Gf.Quatf(*scene_quat))

    # Add objects at original positions
    assets_path = f"{assets_base}/{config['assets_subpath']}"
    for name, pose in layout.items():
        usd_path = get_object_usd_path(task_type, name, assets_path)
        if not Path(usd_path).exists():
            print(f"[WARN] {usd_path} not found, skipping")
            continue

        prim = stage.DefinePrim(f"/World/{name}")
        prim.GetReferences().AddReference(usd_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pose["pos"]))
        xform.AddOrientOp().Set(Gf.Quatf(*pose["rot"]))
        print(f"[OK] Added {name}")

    stage.Save()
    print(f"[OK] Saved: {output_usd}")
    return output_usd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compose USD scene with transformed background and task objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--task", required=True, choices=list(TASK_TO_ENV.keys()), help="Task type")
    parser.add_argument("--background", required=True, metavar="USD", help="Background scene USD")
    parser.add_argument("--output", required=True, metavar="USD", help="Output USD path")
    parser.add_argument("--assets-base", required=True, metavar="DIR", help="Base path for object assets")
    parser.add_argument(
        "--target-pos", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="Target robot position"
    )
    parser.add_argument(
        "--target-quat",
        type=float,
        nargs=4,
        default=[1, 0, 0, 0],
        metavar=("W", "X", "Y", "Z"),
        help="Target robot quaternion",
    )
    parser.add_argument("--include-table", action="store_true", help="Include table in output")
    parser.add_argument(
        "--dual-arm",
        action="store_true",
        help="Use dual-arm configuration (toys/cloth only, uses left_arm as reference)",
    )

    args = parser.parse_args()

    # Set LEISAAC_ASSETS_ROOT to the provided assets base path
    import os

    os.environ["LEISAAC_ASSETS_ROOT"] = args.assets_base
    print(f"[INFO] Set LEISAAC_ASSETS_ROOT={args.assets_base}")

    # Initialize Isaac Sim
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app

    # Register environments
    import leisaac.tasks  # noqa: F401

    # Compose scene
    compose_scene(
        task_type=args.task,
        background_usd=args.background,
        output_usd=args.output,
        assets_base=args.assets_base,
        target_pos=args.target_pos,
        target_quat=args.target_quat,
        include_table=args.include_table,
        use_dual_arm=args.dual_arm,
    )

    simulation_app.close()
