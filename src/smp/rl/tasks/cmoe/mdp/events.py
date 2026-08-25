# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE reset and perturbation events."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def hold_default_joint_targets(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Keep joints outside the 12-DoF action set at their default positions."""
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  target = default if env_ids is None else default[env_ids]
  asset.set_joint_position_target(
    target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
  )


def reset_joints_by_scale(
  env,
  env_ids: torch.Tensor,
  position_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Reset joints to a random scale of their default pose."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  default = asset.data.default_joint_pos[env_ids][:, joint_ids]
  scale = torch.empty(len(env_ids), 1, device=env.device).uniform_(*position_range)
  asset.write_joint_state_to_sim(
    default * scale,
    torch.zeros_like(default),
    joint_ids=joint_ids,
    env_ids=env_ids,
  )


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_motor_parameters(
  env,
  env_ids: torch.Tensor | None,
  motor_range: tuple[float, float],
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply the reset-time Kp/Kd factors used by the original CMoE task."""
  asset: Entity = env.scene[asset_cfg.name]
  env_ids = (
    torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  )
  ctrl_ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]
  motor = torch.empty(len(env_ids), 1, device=env.device).uniform_(*motor_range)
  kp = torch.empty(len(env_ids), 1, device=env.device).uniform_(*kp_range)
  kd = torch.empty(len(env_ids), 1, device=env.device).uniform_(*kd_range)
  gain = env.sim.get_default_field("actuator_gainprm")
  bias = env.sim.get_default_field("actuator_biasprm")
  env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
    gain[ctrl_ids, 0] * kp * motor
  )
  env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
    bias[ctrl_ids, 1] * kp * motor
  )
  env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
    bias[ctrl_ids, 2] * kd * motor
  )


def reset_height_scan(
  env, env_ids: torch.Tensor | slice | None, sensor_name: str
) -> None:
  """Resample the height-scan XY and yaw offsets at every episode reset."""
  env.scene[sensor_name].reset_scan_noise(env_ids)


def apply_external_force_local(
  env,
  env_ids: torch.Tensor,
  force_range: tuple[float, float],
  torque_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply the original local-frame root disturbance and retain its label."""
  del torque_range
  asset: Entity = env.scene[asset_cfg.name]
  if not hasattr(env, "cmoe_external_force"):
    env.cmoe_external_force = torch.zeros(env.num_envs, 3, device=env.device)
  local_force = torch.empty(len(env_ids), 3, device=env.device).uniform_(*force_range)
  env.cmoe_external_force[env_ids] = local_force
  world_force = quat_apply(asset.data.root_link_quat_w[env_ids], local_force)
  asset.write_external_wrench_to_sim(
    world_force[:, None, :],
    torch.zeros_like(world_force[:, None, :]),
    env_ids=env_ids,
    body_ids=asset_cfg.body_ids,
  )


def clear_external_force(
  env,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Clear the previous one-step disturbance."""
  asset: Entity = env.scene[asset_cfg.name]
  if not hasattr(env, "cmoe_external_force"):
    env.cmoe_external_force = torch.zeros(env.num_envs, 3, device=env.device)
  env.cmoe_external_force[:] = 0.0
  asset.write_external_wrench_to_sim(
    torch.zeros(env.num_envs, 1, 3, device=env.device),
    torch.zeros(env.num_envs, 1, 3, device=env.device),
    body_ids=asset_cfg.body_ids,
  )


__all__ = [
  "apply_external_force_local",
  "clear_external_force",
  "hold_default_joint_targets",
  "randomize_motor_parameters",
  "reset_height_scan",
  "reset_joints_by_scale",
]
