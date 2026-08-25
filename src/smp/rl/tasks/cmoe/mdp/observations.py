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

from smp.rl.tasks.cmoe.mdp.events import external_force
from smp.rl.tasks.cmoe.terrain import cmoe_scan_heights

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
    self._reset_envs = torch.zeros(data.nworld, dtype=torch.bool, device=device)
    self._corrupt_reset_envs = torch.zeros(data.nworld, dtype=torch.bool, device=device)
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
    self._reset_envs[env_ids] = True
    self._corrupt_reset_envs[env_ids] = True

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

    root_pos, yaw, world_origins = self._scan_points()
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

  def _scan_points(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert self._data is not None
    assert self._local_offsets is not None
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
    return root_pos, yaw, world_origins

  def terrain_heights(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
    if self._height_step == step:
      assert self._height_observation is not None
      if self._reset_envs.any():
        heights = cmoe_scan_heights(env, self._scan_points()[2])
        self._height_observation[self._reset_envs] = heights[self._reset_envs]
        self._height_history[self._reset_envs] = heights[self._reset_envs]
        self._height_history_valid[self._reset_envs] = True
        self._reset_envs[:] = False
      return self._height_observation
    heights = cmoe_scan_heights(env, self._scan_points()[2])
    if self._height_history_valid.any():
      update = torch.rand(heights.shape[0], device=heights.device) > 0.2
      update |= ~self._height_history_valid
      heights = torch.where(update[:, None], heights, self._height_history)
    self._height_history.copy_(heights)
    self._height_history_valid[:] = True
    self._height_step = step
    self._height_observation = heights
    self._reset_envs[:] = False
    return heights


def cmoe_proprio(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_scale: tuple[float, float, float],
  ang_vel_scale: float,
  joint_pos_scale: float,
  joint_vel_scale: float,
  noise_scales: tuple[float, float, float, float],
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
      command[:, :3] * torch.tensor(command_scale, device=command.device),
      asset.data.root_link_ang_vel_b * ang_vel_scale,
      asset.data.projected_gravity_b,
      (asset.data.joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids])
      * joint_pos_scale,
      (asset.data.joint_vel[:, joint_ids] - default_joint_vel[:, joint_ids])
      * joint_vel_scale,
      env.action_manager.action,
    ),
    dim=-1,
  )
  if not corrupt:
    return obs
  step = int(env.common_step_counter)
  if not hasattr(env, "cmoe_proprio_valid"):
    env.cmoe_proprio_valid = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    env.cmoe_proprio_step = -1
  if env.cmoe_proprio_step != step:
    env.cmoe_proprio_valid[:] = False
    env.cmoe_proprio_step = step
  invalid = ~env.cmoe_proprio_valid
  if invalid.any():
    if not hasattr(env, "cmoe_proprio_observation"):
      env.cmoe_proprio_observation = torch.empty_like(obs)
    env.cmoe_proprio_observation[invalid] = _corrupt_proprio(
      obs[invalid], noise_scales
    )
    env.cmoe_proprio_valid[invalid] = True
  return env.cmoe_proprio_observation


def _corrupt_proprio(
  obs: torch.Tensor, noise_scales: tuple[float, float, float, float]
) -> torch.Tensor:
  """Apply the observation noise used by the original CMoE environment."""
  obs = obs.clone()
  ang_vel, gravity, joint_pos, joint_vel = noise_scales
  obs[:, 3:6] += torch.empty_like(obs[:, 3:6]).uniform_(-ang_vel, ang_vel)
  obs[:, 6:9] += torch.empty_like(obs[:, 6:9]).uniform_(-gravity, gravity)
  obs[:, 9:21] += torch.empty_like(obs[:, 9:21]).uniform_(-joint_pos, joint_pos)
  obs[:, 21:33] += torch.empty_like(obs[:, 21:33]).uniform_(-joint_vel, joint_vel)
  return obs


class ProprioHistory:
  """Keep the newest-first ten-frame 45-D proprioceptive history."""

  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    self.env = env
    self.history = torch.zeros(
      env.num_envs,
      CMOE_HISTORY_LENGTH,
      CMOE_PROPRIO_DIM,
      device=env.device,
    )
    self.step = -1
    self.reset_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_scale: tuple[float, float, float],
    ang_vel_scale: float,
    joint_pos_scale: float,
    joint_vel_scale: float,
    noise_scales: tuple[float, float, float, float],
    corrupt: bool,
  ) -> torch.Tensor:
    current = cmoe_proprio(
      env,
      command_name,
      asset_cfg,
      command_scale,
      ang_vel_scale,
      joint_pos_scale,
      joint_vel_scale,
      noise_scales,
      corrupt,
    )

    step = int(env.common_step_counter)
    repeated = current.unsqueeze(1).expand(-1, CMOE_HISTORY_LENGTH, -1)
    if self.step != step:
      self.history = torch.cat((current.unsqueeze(1), self.history[:, :-1]), dim=1)
      first_step = env.episode_length_buf <= 1
      self.history = torch.where(first_step[:, None, None], repeated, self.history)
      self.step = step
    elif self.reset_envs.any():
      self.history[self.reset_envs] = repeated[self.reset_envs]
    self.reset_envs[:] = False
    return self.history.flatten(start_dim=1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      self.history[:] = 0.0
      self.reset_envs[:] = True
      if hasattr(self.env, "cmoe_proprio_valid"):
        self.env.cmoe_proprio_valid[:] = False
    else:
      self.history[env_ids] = 0.0
      self.reset_envs[env_ids] = True
      if hasattr(self.env, "cmoe_proprio_valid"):
        self.env.cmoe_proprio_valid[env_ids] = False


def cmoe_height_scan(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  height_scale: float,
  noise_amplitude: float,
  extreme_points: int,
  corrupt: bool = True,
) -> torch.Tensor:
  """Return the 77 world-frame terrain heights with CMoE scaling."""
  sensor: CMoERayCastSensor = env.scene[sensor_name]
  heights = sensor.terrain_heights(env, env.common_step_counter)
  heights = heights.view(env.num_envs, 7, 11).transpose(1, 2).flatten(1)
  heights = heights * height_scale
  if not corrupt:
    return heights

  if sensor._height_corrupt_step == env.common_step_counter:
    assert sensor._height_corrupted is not None
    if sensor._corrupt_reset_envs.any():
      reset_envs = sensor._corrupt_reset_envs
      sensor._height_corrupted[reset_envs] = _corrupt_heights(
        heights[reset_envs], noise_amplitude, extreme_points
      )
      sensor._corrupt_reset_envs[:] = False
    return sensor._height_corrupted

  heights = _corrupt_heights(heights, noise_amplitude, extreme_points)
  sensor._height_corrupt_step = env.common_step_counter
  sensor._height_corrupted = heights
  sensor._corrupt_reset_envs[:] = False
  return heights


def _corrupt_heights(
  heights: torch.Tensor, noise_amplitude: float, extreme_points: int
) -> torch.Tensor:
  heights = heights + torch.empty_like(heights).uniform_(
    -noise_amplitude, noise_amplitude
  )
  indices = torch.multinomial(
    torch.ones_like(heights), num_samples=2 * extreme_points, replacement=False
  )
  batch = torch.arange(heights.shape[0], device=heights.device)[:, None]
  row_max = heights.max(dim=1, keepdim=True).values
  row_min = heights.min(dim=1, keepdim=True).values
  high = (
    torch.rand(heights.shape[0], extreme_points, device=heights.device)
    * (row_max - row_min)
    + row_max
  )
  low = (
    torch.rand(heights.shape[0], extreme_points, device=heights.device)
    * (row_min - row_max)
    + row_min
  )
  heights[batch, indices[:, :extreme_points]] = high
  heights[batch, indices[:, extreme_points:]] = low
  return heights


def cmoe_base_lin_vel(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Privileged base linear velocity, scaled as in CMoE."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b * 2.0


def cmoe_external_force(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Privileged external force on the root body."""
  return external_force(env)


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
