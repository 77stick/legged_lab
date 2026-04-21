import os

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.v1 import V1_IMPLICIT_CFG
from legged_lab.tasks.locomotion.animation.animation_env_cfg import AnimationEnvCfg


@configclass
class V1AnimEnvCfg(AnimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot_anim = V1_IMPLICIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_anim")
        self.scene.robot_anim.spawn.rigid_props.disable_gravity = True  # type: ignore
        self.scene.robot_anim.spawn.articulation_props.enabled_self_collisions = False  # type: ignore
        self.scene.robot_anim.spawn.activate_contact_sensors = False  # type: ignore
        self.scene.robot_anim.spawn.collision_props = sim_utils.CollisionPropertiesCfg(  # type: ignore
            collision_enabled=False
        )

        # Playback source: Kimodo-generated + GMR-retargeted V1 motions
        # (17 categories x 3 seeds = 51 clips, flat layout).
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "v1", "amp_dataset"
        )
        # Keys must match .pkl stems under the motion_data_dir above.
        _category_priority = {
            "stand_idle": 1.5,
            "stand_shift_weight": 1.0,
            "walk_forward_very_slow": 1.5,
            "walk_forward_slow": 2.0,
            "walk_forward_mid": 2.0,
            "walk_forward_normal": 2.0,
            "walk_backward_slow": 1.2,
            "side_step_left_slow": 1.2,
            "side_step_right_slow": 1.2,
            "turn_left_in_place": 1.2,
            "turn_right_in_place": 1.2,
            "walk_arc_left_slow": 1.5,
            "walk_arc_right_slow": 1.5,
            "walk_arc_left_normal": 1.2,
            "walk_arc_right_normal": 1.2,
            "walk_diag_left_slow": 1.0,
            "walk_diag_right_slow": 1.0,
        }
        self.motion_data.motion_dataset.motion_data_weights = {
            f"{category}_s{seed:02d}": weight
            for category, weight in _category_priority.items()
            for seed in range(3)
        }

        self.animation.animation.random_initialize = True
        self.animation.animation.num_steps_to_use = 10
