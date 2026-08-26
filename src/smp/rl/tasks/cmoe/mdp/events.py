# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE reset and perturbation events."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def external_force(env) -> torch.Tensor:
  if not hasattr(env, "cmoe_external_force"):
    env.cmoe_external_force = torch.zeros(env.num_envs, 3, device=env.device)
  return env.cmoe_external_force


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
  scale = torch.empty_like(default).uniform_(*position_range)
  asset.write_joint_state_to_sim(
    default * scale,
    torch.zeros_like(default),
    joint_ids=joint_ids,
    env_ids=env_ids,
  )


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def randomize_pd_gains(
  env,
  env_ids: torch.Tensor | None,
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
  kp = torch.empty(len(env_ids), 1, device=env.device).uniform_(*kp_range)
  kd = torch.empty(len(env_ids), 1, device=env.device).uniform_(*kd_range)
  gain = env.sim.get_default_field("actuator_gainprm")
  bias = env.sim.get_default_field("actuator_biasprm")
  env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = gain[ctrl_ids, 0] * kp
  env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = bias[ctrl_ids, 1] * kp
  env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = bias[ctrl_ids, 2] * kd


@requires_model_fields(
  "body_mass",
  "body_ipos",
  recompute=RecomputeLevel.set_const,
)
def randomize_base_inertial_properties(
  env,
  env_ids: torch.Tensor | None,
  payload_range: tuple[float, float],
  com_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Randomize the base mass and COM."""
  env_ids = (
    torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  )
  body_ids = torch.as_tensor(asset_cfg.body_ids, device=env.device)
  mass = env.sim.get_default_field("body_mass")[body_ids]
  payload = torch.empty(len(env_ids), len(body_ids), device=env.device).uniform_(
    *payload_range
  )
  randomized_mass = mass + payload
  env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
  env.sim.model.body_mass[env_grid, body_grid] = randomized_mass
  env.sim.model.body_ipos[env_grid, body_grid] = torch.empty(
    len(env_ids), len(body_ids), 3, device=env.device
  ).uniform_(*com_range)


def reset_height_scan(
  env, env_ids: torch.Tensor | slice | None, sensor_name: str
) -> None:
  """Resample the height-scan XY and yaw offsets at every episode reset."""
  env.scene[sensor_name].reset_scan_noise(env_ids)


def push_robot(
  env,
  env_ids: torch.Tensor | None,
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Set the root XY velocity to the original CMoE push sample."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  velocity = asset.data.root_link_vel_w[env_ids].clone()
  velocity[:, :2] = torch.empty(len(env_ids), 2, device=env.device).uniform_(
    *velocity_range
  )
  asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


def apply_external_force_local(
  env,
  env_ids: torch.Tensor | None,
  force_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply the original local-frame root disturbance and retain its label."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  local_force = torch.empty(len(env_ids), 3, device=env.device).uniform_(*force_range)
  external_force(env)[env_ids] = local_force
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
  external_force(env)[:] = 0.0
  asset.write_external_wrench_to_sim(
    torch.zeros(env.num_envs, 1, 3, device=env.device),
    torch.zeros(env.num_envs, 1, 3, device=env.device),
    body_ids=asset_cfg.body_ids,
  )


def reset_external_force(
  env,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  external_force(env)[env_ids] = 0.0
  zeros = torch.zeros(len(env_ids), 1, 3, device=env.device)
  asset.write_external_wrench_to_sim(
    zeros,
    zeros,
    env_ids=env_ids,
    body_ids=asset_cfg.body_ids,
  )


__all__ = [
  "apply_external_force_local",
  "clear_external_force",
  "external_force",
  "push_robot",
  "randomize_base_inertial_properties",
  "randomize_pd_gains",
  "reset_external_force",
  "reset_height_scan",
  "reset_joints_by_scale",
]
