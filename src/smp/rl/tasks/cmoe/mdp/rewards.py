# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE locomotion rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


def _asset(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def tracking_lin_vel(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  command = _command(env, command_name)
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-error / sigma)


def tracking_ang_vel(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sigma: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  command = _command(env, command_name)
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / sigma)


def tracking_yaw(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track the commanded heading, as in the original CMoE reward."""
  asset = _asset(env, asset_cfg)
  command_term = env.command_manager.get_term(command_name)
  error = wrap_to_pi(command_term.heading_target - asset.data.heading_w)
  return torch.exp(-error.abs())


def lin_vel_z(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  return torch.square(_asset(env, asset_cfg).data.root_link_lin_vel_b[:, 2])


def ang_vel_xy(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  return torch.sum(
    torch.square(_asset(env, asset_cfg).data.root_link_ang_vel_b[:, :2]), dim=1
  )


def orientation(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  gravity = _asset(env, asset_cfg).data.projected_gravity_b
  return torch.sum(torch.square(gravity[:, :2]), dim=1)


def base_height(
  env: "ManagerBasedRlEnv",
  contact_sensor_name: str,
  foot_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  contact = _contact_filter(env, contact_sensor_name).float()
  count = contact.sum(dim=1)
  feet_z = torch.sum(feet * contact, dim=1) / count.clamp_min(1.0)
  feet_z = torch.where(count > 0, feet_z, feet.mean(dim=1))
  height = asset.data.root_link_pos_w[:, 2] - (feet_z - foot_height)
  return torch.square(height - 0.75)


def feet_stumble(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return torch.any(
    torch.linalg.norm(force[..., :2], dim=-1) > 3.0 * force[..., 2].abs(), dim=1
  ).float()


def collision(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return (torch.linalg.norm(force, dim=-1) > 0.1).float().sum(dim=1)


def feet_lateral_distance(
  env: "ManagerBasedRlEnv",
  min_distance: float,
  max_distance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids]
  root_pos = asset.data.root_link_pos_w.unsqueeze(1)
  root_quat = asset.data.root_link_quat_w[:, None].expand(
    -1, len(asset_cfg.site_ids), -1
  )
  feet_b = quat_apply_inverse(
    root_quat.reshape(-1, 4), (feet - root_pos).reshape(-1, 3)
  ).view_as(feet)
  distance = (feet_b[:, 0, 1] - feet_b[:, 1, 1]).abs()
  return (distance - min_distance).clamp(max=max_distance - min_distance)


def feet_air_time(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float | None = None,
) -> torch.Tensor:
  del command_name, command_threshold
  contact = _contact_filter(env, sensor_name)
  if not hasattr(env, "cmoe_feet_air_time"):
    env.cmoe_feet_air_time = torch.zeros(
      env.num_envs, contact.shape[1], device=env.device
    )
  reset = env.episode_length_buf <= 1
  env.cmoe_feet_air_time[reset] = 0.0
  first_contact = (env.cmoe_feet_air_time > 0.0) & contact
  env.cmoe_feet_air_time += env.step_dt
  reward = torch.sum((env.cmoe_feet_air_time - 0.5) * first_contact, dim=1)
  env.cmoe_feet_air_time *= ~contact
  return reward


def feet_slip(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  contact = sensor.data.found
  assert contact is not None
  asset = _asset(env, SceneEntityCfg("robot"))
  foot_ids = asset.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
  speed = torch.linalg.norm(asset.data.site_lin_vel_w[:, foot_ids], dim=-1)
  return torch.sum(speed * (contact > 0).float(), dim=1)


def feet_ground_parallel(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: TerrainHeightSensor = env.scene[sensor_name]
  heights = sensor.data.heights
  return torch.var(heights[:, :5], dim=-1) + torch.var(heights[:, 5:], dim=-1)


def feet_edge(
  env: "ManagerBasedRlEnv",
  height_sensor_name: str,
  contact_sensor_name: str,
  edge_threshold: float = 0.05,
) -> torch.Tensor:
  height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
  heights = height_sensor.data.heights.view(env.num_envs, 2, 5)
  at_edge = heights.amax(dim=-1) - heights.amin(dim=-1) > edge_threshold
  in_contact = _contact_filter(env, contact_sensor_name)
  terrain = env.scene.terrain
  exclude = (terrain.terrain_types >= 4) & (terrain.terrain_types <= 15)
  reward = (at_edge & in_contact).sum(dim=-1) * (terrain.terrain_levels > 3)
  return reward * ~exclude


def hip_dof_error(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  default = asset.data.default_joint_pos
  assert default is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(error), dim=1)


def dof_acc(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return torch.sum(
    torch.square(_asset(env, asset_cfg).data.joint_acc[:, asset_cfg.joint_ids]), dim=1
  )


def dof_vel(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return torch.sum(
    torch.square(_asset(env, asset_cfg).data.joint_vel[:, asset_cfg.joint_ids]), dim=1
  )


def torques(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return torch.sum(
    torch.square(_asset(env, asset_cfg).data.qfrc_actuator[:, asset_cfg.joint_ids]),
    dim=1,
  )


def action_rate(env: "ManagerBasedRlEnv") -> torch.Tensor:
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def dof_pos_limits(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  lower = -(pos - limits[..., 0]).clamp(max=0.0)
  upper = (pos - limits[..., 1]).clamp(min=0.0)
  return torch.sum(lower + upper, dim=1)


def dof_vel_limits(
  env: "ManagerBasedRlEnv",
  velocity_limits: tuple[float, ...],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  limits = torch.tensor(velocity_limits, device=env.device)
  excess = (asset.data.joint_vel[:, asset_cfg.joint_ids].abs() - limits).clamp(
    min=0.0, max=1.0
  )
  return torch.sum(excess, dim=1)


def torque_limits(
  env: "ManagerBasedRlEnv",
  torque_limits: tuple[float, ...],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  limits = torch.tensor(torque_limits, device=env.device)
  excess = (asset.data.qfrc_actuator[:, asset_cfg.joint_ids].abs() - limits).clamp(
    min=0.0
  )
  return torch.sum(excess, dim=1)


tracking_linear_velocity = tracking_lin_vel
tracking_angular_velocity = tracking_ang_vel


def _contact_filter(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  """Return current contact OR previous contact, matching CMoE's filter."""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  if force is not None:
    contact = torch.linalg.norm(force, dim=-1) > 2.0
  else:
    found = sensor.data.found
    assert found is not None
    contact = found > 0
  if not hasattr(env, "cmoe_contact_current"):
    env.cmoe_contact_current = torch.zeros_like(contact)
    env.cmoe_contact_previous = torch.zeros_like(contact)
    env.cmoe_contact_step = -1
  step = int(env.common_step_counter)
  if env.cmoe_contact_step != step:
    env.cmoe_contact_previous.copy_(env.cmoe_contact_current)
    env.cmoe_contact_previous[env.episode_length_buf <= 1] = False
    env.cmoe_contact_current.copy_(contact)
    env.cmoe_contact_step = step
  return env.cmoe_contact_current | env.cmoe_contact_previous


__all__ = [
  "action_rate",
  "ang_vel_xy",
  "base_height",
  "collision",
  "dof_acc",
  "dof_pos_limits",
  "dof_vel",
  "dof_vel_limits",
  "feet_air_time",
  "feet_edge",
  "feet_ground_parallel",
  "feet_lateral_distance",
  "feet_slip",
  "feet_stumble",
  "hip_dof_error",
  "lin_vel_z",
  "orientation",
  "torque_limits",
  "torques",
  "tracking_ang_vel",
  "tracking_angular_velocity",
  "tracking_lin_vel",
  "tracking_linear_velocity",
  "tracking_yaw",
]
