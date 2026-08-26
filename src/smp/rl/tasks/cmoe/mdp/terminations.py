# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE termination terms."""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from smp.rl.tasks.cmoe.terrain import cmoe_terrain_class


def time_out(env) -> torch.Tensor:
  return env.episode_length_buf > env.max_episode_length


def pelvis_contact(env, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  return torch.any(torch.linalg.norm(force, dim=-1) > 1.0, dim=1)


def bad_orientation(env, limit_angle: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Terminate when either roll or pitch exceeds the original cutoff."""
  asset: Entity = env.scene[asset_cfg.name]
  w, x, y, z = asset.data.root_link_quat_w.unbind(dim=-1)
  roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
  pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
  return (roll.abs() > limit_angle) | (pitch.abs() > limit_angle)


def root_height_below_on_terrain(env, minimum_height: float) -> torch.Tensor:
  robot = env.scene["robot"]
  return (robot.data.root_link_pos_w[:, 2] < minimum_height) & (
    cmoe_terrain_class(env) == 5
  )


__all__ = [
  "bad_orientation",
  "pelvis_contact",
  "root_height_below_on_terrain",
  "time_out",
]
