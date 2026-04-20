"""One-shot script to convert v1 URDF to USD via Isaac Lab's UrdfConverter."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert v1 URDF to USD.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ROBOT_DIR = os.path.join(REPO_ROOT, "source", "legged_lab", "legged_lab", "data", "Robots", "v1")

cfg = UrdfConverterCfg(
    asset_path=os.path.join(ROBOT_DIR, "v1.urdf"),
    usd_dir=os.path.join(ROBOT_DIR, "usd"),
    usd_file_name="v1.usd",
    force_usd_conversion=True,
    make_instanceable=False,
    fix_base=False,
    merge_fixed_joints=False,
    self_collision=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=100.0,
            damping=10.0,
        ),
    ),
)

converter = UrdfConverter(cfg)
print(f"USD file generated at: {converter.usd_path}")

app_launcher.app.close()
