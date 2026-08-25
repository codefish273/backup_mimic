# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""The fixed-waist Unitree G1 asset used by the original CMoE task."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_G1_URDF = (
  Path(__file__).parent
  / "assets"
  / "g1"
  / "29dof_urdf"
  / "g1_29dof_with_hand_fixed_modify_collision.urdf"
)
_G1_MESHES = _G1_URDF.parent.parent / "meshes"

LOWER_BODY_JOINTS = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)


def get_cmoe_g1_spec() -> mujoco.MjSpec:
  """Load the 12-DoF URDF and add the CMoE sensor attachment sites."""
  spec = mujoco.MjSpec.from_file(str(_G1_URDF))
  spec.compiler.meshdir = str(_G1_MESHES)
  spec.body("pelvis").add_freejoint(name="freejoint")

  collision_index = 0
  for geom in spec.geoms:
    if geom.contype:
      geom.name = f"{geom.parent.name}_collision_{collision_index}"
      collision_index += 1

  spec.body("pelvis").add_site(name="cmoe_scan_frame", pos=(0.4, 0.0, 0.0))
  sample_positions = (
    (0.03, 0.0, -0.035),
    (0.12, 0.0, -0.035),
    (-0.05, 0.0, -0.035),
    (0.06, 0.03, -0.035),
    (0.06, -0.03, -0.035),
  )
  for side in ("left", "right"):
    foot = spec.body(f"{side}_ankle_roll_link")
    foot.add_site(name=f"{side}_foot", pos=(0.0, 0.0, 0.0))
    for index, position in enumerate(sample_positions, 1):
      foot.add_site(name=f"cmoe_{side}_foot_sample_point{index}", pos=position)
  return spec


def get_cmoe_g1_robot_cfg() -> EntityCfg:
  """Return the original CMoE G1: fixed waist/arms and 12 actuated joints."""
  actuators = (
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
      stiffness=100.0,
      damping=2.0,
      effort_limit=88.0,
      delay_min_lag=0,
      delay_max_lag=3,
      delay_update_period=4,
      delay_per_env_phase=False,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_roll_joint",),
      stiffness=100.0,
      damping=2.0,
      effort_limit=139.0,
      delay_min_lag=0,
      delay_max_lag=3,
      delay_update_period=4,
      delay_per_env_phase=False,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_knee_joint",),
      stiffness=150.0,
      damping=4.0,
      effort_limit=139.0,
      delay_min_lag=0,
      delay_max_lag=3,
      delay_update_period=4,
      delay_per_env_phase=False,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
      stiffness=40.0,
      damping=2.0,
      effort_limit=50.0,
      delay_min_lag=0,
      delay_max_lag=3,
      delay_update_period=4,
      delay_per_env_phase=False,
    ),
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.8),
      joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_cmoe_g1_spec,
    collisions=(
      CollisionCfg(
        geom_names_expr=(r".*_collision_.*",),
        condim={r"^(left|right)_ankle_roll_link_collision_.*$": 3, ".*": 3},
      ),
    ),
    articulation=EntityArticulationInfoCfg(actuators=actuators),
  )


__all__ = ["LOWER_BODY_JOINTS", "get_cmoe_g1_robot_cfg", "get_cmoe_g1_spec"]
