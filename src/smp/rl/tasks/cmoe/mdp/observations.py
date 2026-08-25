# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE proprioception and terrain observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp
from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.raycast_sensor import RayCastSensor, RayCastSensorCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


CMOE_PROPRIO_DIM = 45
CMOE_HISTORY_LENGTH = 10
CMOE_HEIGHT_SCAN_DIM = 77


@dataclass(kw_only=True)
class CMoERayCastSensorCfg(RayCastSensorCfg):
  """Raycast configuration with the reset-time CMoE scan perturbation."""

  def build(self) -> "CMoERayCastSensor":
    return CMoERayCastSensor(self)


class CMoERayCastSensor(RayCastSensor):
  """Sample the height map at the original reset-randomized CMoE points."""

  cfg: CMoERayCastSensorCfg

  def initialize(self, mj_model, model, data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    self._xy_noise = torch.zeros(data.nworld, 2, device=device)
    self._yaw_noise = torch.zeros(data.nworld, device=device)
    self._height_history = torch.zeros(data.nworld, self.num_rays, device=device)
    self._height_history_valid = torch.zeros(
      data.nworld, dtype=torch.bool, device=device
    )
    self._height_step = -1
    self._height_observation: torch.Tensor | None = None
    self._height_corrupt_step = -1
    self._height_corrupted: torch.Tensor | None = None
    self._frame_local_pos = torch.stack(
      [
        torch.as_tensor(mj_model.site_pos[sid], device=device, dtype=torch.float32)
        for kind, sid, _ in self._frame_infos
      ]
    )
    self.reset_scan_noise()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._height_history[env_ids] = 0.0
    self._height_history_valid[env_ids] = False
    self._height_step = -1
    self._height_corrupt_step = -1

  def reset_scan_noise(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
      count = self._xy_noise.shape[0]
    elif isinstance(env_ids, slice):
      count = len(range(*env_ids.indices(self._xy_noise.shape[0])))
    else:
      count = len(env_ids)
    self._xy_noise[env_ids] = torch.randn(count, 2, device=self._xy_noise.device) * 0.05
    self._yaw_noise[env_ids] = torch.empty(
      count, device=self._yaw_noise.device
    ).uniform_(-0.2, 0.2)
    self._height_history_valid[env_ids] = False

  def prepare_rays(self) -> None:
    super().prepare_rays()
    assert self._data is not None
    assert self._local_offsets is not None and self._local_directions is not None
    assert self._ray_pnt is not None and self._ray_vec is not None
    assert self._frame_local_pos is not None

    body_ids = [body_id for _, _, body_id in self._frame_infos]
    root_pos = self._data.xpos[:, body_ids[0]]
    local_offsets = self._local_offsets + self._frame_local_pos[0]
    c, s = self._yaw_noise.cos(), self._yaw_noise.sin()
    yaw = torch.zeros(self._yaw_noise.shape[0], 3, 3, device=self._yaw_noise.device)
    yaw[:, 0, 0] = c
    yaw[:, 0, 1] = -s
    yaw[:, 1, 0] = s
    yaw[:, 1, 1] = c
    yaw[:, 2, 2] = 1.0
    world_origins = root_pos[:, None, :] + torch.einsum(
      "bij,nj->bni", yaw, local_offsets
    )
    world_origins[:, :, :2] += self._xy_noise[:, None, :]
    world_rays = torch.einsum("bij,nj->bni", yaw, self._local_directions)
    self._cached_world_origins = world_origins
    self._cached_world_rays = world_rays
    self._cached_frame_pos = root_pos[:, None, :] + torch.einsum(
      "bij,fj->bfi", yaw, self._frame_local_pos
    )
    self._cached_frame_mat = yaw[:, None]
    wp.to_torch(self._ray_pnt).view(root_pos.shape[0], self.num_rays, 3).copy_(
      world_origins
    )
    wp.to_torch(self._ray_vec).view(root_pos.shape[0], self.num_rays, 3).copy_(
      world_rays
    )

  def terrain_heights(self, step: int) -> torch.Tensor:
    if self._height_step == step:
      assert self._height_observation is not None
      return self._height_observation
    heights = self.data.hit_pos_w[..., 2] * 5.0
    if self._height_history_valid.any():
      update = torch.rand(heights.shape[0], device=heights.device) > 0.2
      update |= ~self._height_history_valid
      heights = torch.where(update[:, None], heights, self._height_history)
    self._height_history.copy_(heights)
    self._height_history_valid[:] = True
    self._height_step = step
    self._height_observation = heights
    return heights


def cmoe_proprio(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg,
  corrupt: bool = True,
) -> torch.Tensor:
  """Return the 45-dimensional CMoE one-step proprioception vector."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  joint_ids = asset_cfg.joint_ids
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_pos is not None and default_joint_vel is not None
  obs = torch.cat(
    (
      command[:, :3] * torch.tensor((2.0, 2.0, 0.25), device=command.device),
      asset.data.root_link_ang_vel_b * 0.25,
      asset.data.projected_gravity_b,
      asset.data.joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids],
      (asset.data.joint_vel[:, joint_ids] - default_joint_vel[:, joint_ids]) * 0.05,
      env.action_manager.action,
    ),
    dim=-1,
  )
  if not corrupt:
    return obs
  step = int(env.common_step_counter)
  if getattr(env, "cmoe_proprio_step", -1) == step:
    return env.cmoe_proprio_observation
  env.cmoe_proprio_step = step
  env.cmoe_proprio_observation = _corrupt_proprio(obs)
  return env.cmoe_proprio_observation


def _corrupt_proprio(obs: torch.Tensor) -> torch.Tensor:
  """Apply the observation noise used by the original CMoE environment."""
  obs = obs.clone()
  obs[:, 3:6] += torch.empty_like(obs[:, 3:6]).uniform_(-0.05, 0.05)
  obs[:, 6:9] += torch.empty_like(obs[:, 6:9]).uniform_(-0.05, 0.05)
  obs[:, 9:21] += torch.empty_like(obs[:, 9:21]).uniform_(-0.01, 0.01)
  obs[:, 21:33] += torch.empty_like(obs[:, 21:33]).uniform_(-0.075, 0.075)
  return obs


class ProprioHistory:
  """Keep the newest-first ten-frame 45-D proprioceptive history."""

  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    self.history = torch.zeros(
      env.num_envs,
      CMOE_HISTORY_LENGTH,
      CMOE_PROPRIO_DIM,
      device=env.device,
    )

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    corrupt: bool,
  ) -> torch.Tensor:
    current = cmoe_proprio(env, command_name, asset_cfg, corrupt=corrupt)

    self.history = torch.cat((current.unsqueeze(1), self.history[:, :-1]), dim=1)
    first_step = env.episode_length_buf <= 1
    repeated = current.unsqueeze(1).expand(-1, CMOE_HISTORY_LENGTH, -1)
    self.history = torch.where(first_step[:, None, None], repeated, self.history)
    return self.history.flatten(start_dim=1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self.history[:] = 0.0
    else:
      self.history[env_ids] = 0.0


def cmoe_height_scan(
  env: "ManagerBasedRlEnv", sensor_name: str, corrupt: bool = True
) -> torch.Tensor:
  """Return the 77 world-frame terrain heights with CMoE scaling."""
  sensor: RayCastSensor = env.scene[sensor_name]
  heights = (
    sensor.terrain_heights(env.common_step_counter)
    if isinstance(sensor, CMoERayCastSensor)
    else sensor.data.hit_pos_w[..., 2] * 5.0
  )
  heights = heights.view(env.num_envs, 7, 11).transpose(1, 2).flatten(1)
  if not corrupt:
    return heights

  if (
    isinstance(sensor, CMoERayCastSensor)
    and sensor._height_corrupt_step == env.common_step_counter
  ):
    assert sensor._height_corrupted is not None
    return sensor._height_corrupted

  heights = heights + torch.empty_like(heights).uniform_(-0.15, 0.15)
  indices = torch.multinomial(
    torch.ones_like(heights), num_samples=8, replacement=False
  )
  batch = torch.arange(env.num_envs, device=env.device)[:, None]
  row_max = heights.max(dim=1, keepdim=True).values
  row_min = heights.min(dim=1, keepdim=True).values
  high = torch.rand(env.num_envs, 4, device=env.device) * (row_max - row_min) + row_max
  low = torch.rand(env.num_envs, 4, device=env.device) * (row_min - row_max) + row_min
  heights[batch, indices[:, :4]] = high
  heights[batch, indices[:, 4:]] = low
  if isinstance(sensor, CMoERayCastSensor):
    sensor._height_corrupt_step = env.common_step_counter
    sensor._height_corrupted = heights
  return heights


def cmoe_base_lin_vel(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Privileged base linear velocity, scaled as in CMoE."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b * 2.0


def cmoe_external_force(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Privileged external force on the root body."""
  del asset_cfg
  if not hasattr(env, "cmoe_external_force"):
    env.cmoe_external_force = torch.zeros(env.num_envs, 3, device=env.device)
  return env.cmoe_external_force


__all__ = [
  "CMoERayCastSensor",
  "CMoERayCastSensorCfg",
  "CMOE_HEIGHT_SCAN_DIM",
  "CMOE_HISTORY_LENGTH",
  "CMOE_PROPRIO_DIM",
  "ProprioHistory",
  "cmoe_base_lin_vel",
  "cmoe_external_force",
  "cmoe_height_scan",
  "cmoe_proprio",
]
