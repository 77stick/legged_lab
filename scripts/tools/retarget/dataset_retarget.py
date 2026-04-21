"""
Batch retargeting tool: convert multiple GMR motion files to Legged Lab format.

Behavior:
 - Reads all .pkl files from the input directory (sorted).
 - For each file, loads the GMR pickle and uses extract_gmr_data to convert it (entire motion).
 - Runs the simulator once with all motions (num_envs = number of motions) and collects key body positions.
 - Saves each converted motion dict to the output directory with the same filename.

Usage example:
    python scripts/tools/retarget/dataset_retarget.py \
        --robot g1 \
        --input_dir data/gmr/ \
        --output_dir data/lab/ \
        --config_file scripts/tools/retarget/config/g1_29dof.yaml \
        --loop clamp

This script intentionally does NOT support start/end frame clipping; it converts full motions.
"""

import argparse
import json
import pickle
import warnings
import yaml
from pathlib import Path

from isaaclab.app import AppLauncher

# append AppLauncher cli args
parser = argparse.ArgumentParser(description="Batch retarget GMR -> Legged Lab (multiple files).")
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    help="Robot name to use (default: g1)",
)
parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Directory containing input GMR .pkl files",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to write converted .pkl files",
)
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    help="Path to YAML config containing gmr_dof_names, lab_dof_names, lab_key_body_names",
)
parser.add_argument(
    "--loop",
    type=str,
    choices=["wrap", "clamp"],
    default="clamp",
    help="Loop mode for motion (default: clamp)",
)
parser.add_argument(
    "--manifest",
    type=str,
    default=None,
    help=(
        "Optional manifest.json path. When provided, the script reads the clip list from the manifest, "
        "writes flat output files (named by clip 'name'), and bakes per-clip metadata "
        "(cmd_lin_vel_x/y, cmd_ang_vel_z, category, priority) into each output pickle."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

"""Launch Omniverse App"""
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import sys

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

# load robot cfg as single_retarget does
if args_cli.robot == "g1":
    from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
elif args_cli.robot == "v1":
    from legged_lab.assets.v1 import V1_CFG as ROBOT_CFG
else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

# import functions from gmr_to_lab (must be in same directory)
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
try:
    from gmr_to_lab import LoopMode, ReplayMotionsSceneCfg, extract_gmr_data, run_simulator
except ImportError as e:
    print(f"Error importing from gmr_to_lab.py: {e}")
    raise


def list_input_files(input_dir: str):
    p = Path(input_dir)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix == ".pkl"])
    return files


def list_input_files_from_manifest(input_dir: str, manifest_path: str):
    """Read clip list from a manifest.json and resolve .pkl paths under input_dir.

    Returns a list of (Path, output_name, clip_dict) tuples, one per clip.
    The clip ``path`` field in the manifest (originally .npz) is remapped to .pkl.
    """
    in_root = Path(input_dir)
    with open(manifest_path) as f:
        manifest = json.load(f)
    clips = manifest.get("clips", [])
    if not clips:
        raise ValueError(f"Manifest has no 'clips': {manifest_path}")

    resolved: list[tuple[Path, str, dict]] = []
    missing: list[str] = []
    for clip in clips:
        rel = Path(clip["path"]).with_suffix(".pkl")
        pkl_path = in_root / rel
        if not pkl_path.exists():
            missing.append(str(rel))
            continue
        out_name = f"{clip['name']}.pkl"
        resolved.append((pkl_path, out_name, clip))

    if missing:
        warnings.warn(f"{len(missing)} manifest clips missing .pkl under {in_root}: {missing[:5]}...")
    return resolved


def main():
    # read config
    with open(args_cli.config_file) as f:
        config = yaml.safe_load(f)

    gmr_dof_names = config["gmr_dof_names"]
    lab_dof_names = config["lab_dof_names"]
    lab_key_body_names = config["lab_key_body_names"]

    loop_mode = LoopMode.CLAMP if args_cli.loop == "clamp" else LoopMode.WRAP

    if args_cli.manifest:
        resolved = list_input_files_from_manifest(args_cli.input_dir, args_cli.manifest)
        if len(resolved) == 0:
            print(f"No clips resolved from manifest: {args_cli.manifest}")
            return
        input_files = [p for p, _, _ in resolved]
        output_names = [name for _, name, _ in resolved]
        clip_meta = [meta for _, _, meta in resolved]
        print(f"Found {len(input_files)} manifest clips under {args_cli.input_dir} (flat output).")
    else:
        input_files = list_input_files(args_cli.input_dir)
        if len(input_files) == 0:
            print(f"No .pkl files found in input directory: {args_cli.input_dir}")
            return
        output_names = [p.name for p in input_files]
        clip_meta = [None] * len(input_files)
        print(f"Found {len(input_files)} files to convert.")

    Path(args_cli.output_dir).mkdir(parents=True, exist_ok=True)

    # load and convert all gmr files (entire motion)
    motion_data_dicts = []
    fps_values = []

    for p in input_files:
        print(f"Loading and converting: {p.name}")
        motion = extract_gmr_data(
            gmr_file_path=str(p),
            gmr_dof_names=gmr_dof_names,
            lab_dof_names=lab_dof_names,
            loop_mode=loop_mode,
            start_frame=0,
            end_frame=-1,
        )
        motion_data_dicts.append(motion)
        fps_values.append(motion["fps"])

    # check fps consistency
    if not all(f == fps_values[0] for f in fps_values):
        print(fps_values)
        warnings.warn("Motions have different fps. Using fps from first motion.")

    fps = fps_values[0]
    dt = 1.0 / fps

    # start simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args_cli.device))
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=len(motion_data_dicts),
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    scene = InteractiveScene(scene_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    sim.reset()
    print("Simulation starting ...")

    # run simulator with all motions
    motion_data_dicts = run_simulator(simulation_app, sim, scene, motion_data_dicts, lab_key_body_names)

    # save outputs
    print("Saving converted motions to output directory...")
    for name, motion, meta in zip(output_names, motion_data_dicts, clip_meta):
        if meta is not None:
            motion["cmd_lin_vel_x"] = float(meta.get("cmd_lin_vel_x", 0.0))
            motion["cmd_lin_vel_y"] = float(meta.get("cmd_lin_vel_y", 0.0))
            motion["cmd_ang_vel_z"] = float(meta.get("cmd_ang_vel_z", 0.0))
            motion["category"] = meta.get("category", "")
            motion["priority"] = float(meta.get("priority", 1.0))
            motion["clip_name"] = meta.get("name", Path(name).stem)
        out_path = Path(args_cli.output_dir) / name
        with open(out_path, "wb") as f:
            pickle.dump(motion, f)
        print(f"Saved: {out_path}")

    print("Closing simulation app...")
    simulation_app.close()
    print("Done.")


if __name__ == "__main__":
    main()
