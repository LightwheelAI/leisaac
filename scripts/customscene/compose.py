#!/usr/bin/env python3
"""Scene composition CLI tool for Isaac Sim (Headless)"""

import argparse
from isaaclab.app import AppLauncher

# Default robot poses for each task
# # This is for single-arm setups and the left arm in dual-arm setups.
ROBOT_DEFAULTS = {
    "toys": {"pos": [-0.42, -0.26, 0.43], "quat": [0.0, 0.0, 0.0, 1.0]},
    "orange": {"pos": [2.2, -0.61, 0.89], "quat": [0.0, 0.0, 0.0, 1.0]},
    "cloth": {"pos": [-0.86, 8.35, 3.25], "quat": [0.0, 0.0, 0.0, 1.0]},
    "cube": {"pos": [0.35, -0.36, 0.06], "quat": [0.0, 0.0, 0.0, 1.0]},
}

# Default table poses
TABLE_DEFAULTS = {
    "toys": {"pos": [-0.3864, 0.0, 0.2781], "quat": [1.0, 0.0, 0.0, 0.0]},
    "cloth": {"pos": [-0.8780, 8.732, 2.7574], "quat": [1.0, 0.0, 0.0, 0.0]},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USD Scene Composition", epilog="Example: python compose.py --task toys --background scene.usd --output out.usd --target-pos 0 0 1 --target-quat 1 0 0 0", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument("--task", required=True, choices=list(ROBOT_DEFAULTS.keys()), metavar="TASK", help="Task type: toys, orange, cloth, cube")
    required.add_argument("--background", required=True, metavar="USD", help="Path to background USD scene file")
    required.add_argument("--output", required=True, metavar="USD", help="Path to output composed scene file")
    required.add_argument("--table", action="store_true", help="Use table as reference frame (include table in scene)")
    
    # Optional arguments
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("--json", metavar="PATH", help="Export transformed poses to JSON file")
    optional.add_argument("--assets-base", metavar="DIR", help="Override default assets base directory")
    
    # Transform arguments
    transform = parser.add_argument_group('transform parameters')
    transform.add_argument("--target-pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=[-0.42, -0.26, 0.43], help="Target reference position (default: -0.42 -0.26 0.43)")
    transform.add_argument("--target-quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"), default=[0.0, 0.0, 0.0, 1.0], help="Target reference quaternion (default: 0 0 0 1)")
    
    args = parser.parse_args()
    
    # Initialize Isaac Sim
    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app
    
    
    from Scene_update import compose_scene, transform_to_json, TASK_CONFIG
    
    # Update assets base if provided
    if args.assets_base:
        base = args.assets_base
        TASK_CONFIG["toys"]["assets_path"] = f"{base}/scenes/lightwheel_toyroom/Assets"
        TASK_CONFIG["orange"]["assets_path"] = f"{base}/scenes/kitchen_with_orange/objects"
        TASK_CONFIG["cloth"]["assets_path"] = f"{base}/scenes/lightwheel_bedroom"
        TASK_CONFIG["cube"]["assets_path"] = f"{base}/scenes/table_with_cube/cube"
    
    # Determine reference frame
    include_table = args.table
    robot_default = ROBOT_DEFAULTS[args.task]
    
    if include_table:
        # Table mode: use table as reference, robot position will be transformed
        if args.task not in TABLE_DEFAULTS:
            print(f"[WARN] --table not supported for '{args.task}', using arm reference")
            include_table = False
            orig_pos = robot_default["pos"]
            orig_quat = robot_default["quat"]
        else:
            table_default = TABLE_DEFAULTS[args.task]
            orig_pos = table_default["pos"]
            orig_quat = table_default["quat"]
            print(f"[INFO] Using table reference: {orig_pos}")
    else:
        # Arm mode: use robot as reference
        orig_pos = robot_default["pos"]
        orig_quat = robot_default["quat"]
    
    # Compose scene
    compose_scene(
        task_type=args.task,
        background_usd=args.background,
        output_usd=args.output,
        orig_pos=orig_pos,
        orig_quat=orig_quat,
        target_pos=args.target_pos,
        target_quat=args.target_quat,
        include_table=include_table,
    )
    
    # Export JSON if requested
    if args.json:
        transform_to_json(
            task_type=args.task,
            orig_pos=orig_pos,
            orig_quat=orig_quat,
            target_pos=args.target_pos,
            target_quat=args.target_quat,
            include_table=include_table,
            robot_orig_pos=robot_default["pos"] if include_table else None,
            robot_orig_quat=robot_default["quat"] if include_table else None,
            output_path=args.json,
        )
    
    simulation_app.close()
    print("Done.")
