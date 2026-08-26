# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE curricula."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def terrain_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> dict[str, torch.Tensor]:
  terrain = env.scene.terrain
  levels = terrain.terrain_levels.float()
  if env.common_step_counter == 0:
    return {"mean": levels.mean(), "max": levels.max()}

  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )
  threshold = command[env_ids, 0] * env.max_episode_length_s
  move_up = distance > 0.8 * threshold
  move_down = distance < 0.4 * threshold
  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  return {"mean": levels.mean(), "max": levels.max()}


__all__ = ["terrain_levels"]
