# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE velocity commands."""

from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class CMoEVelocityCommand(UniformVelocityCommand):
  """Use omnidirectional commands on easy terrain and forward commands elsewhere."""

  cfg: "CMoEVelocityCommandCfg"

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    hard_ids = env_ids[env_ids >= self._env.num_envs * 0.1]
    sample = torch.empty(len(hard_ids), device=self.device)
    self.vel_command_b[hard_ids, 0] = sample.uniform_(*self.cfg.hard_ranges.lin_vel_x)
    self.vel_command_b[hard_ids, 1] = sample.uniform_(*self.cfg.hard_ranges.lin_vel_y)
    self.vel_command_b[hard_ids, 2] = sample.uniform_(*self.cfg.hard_ranges.ang_vel_z)
    self.heading_target[hard_ids] = sample.uniform_(*self.cfg.hard_ranges.heading)
    self.is_heading_env[hard_ids] = True
    self.vel_command_b[env_ids, :2] *= (
      torch.linalg.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2
    ).unsqueeze(1)


@dataclass(kw_only=True)
class CMoEVelocityCommandCfg(UniformVelocityCommandCfg):
  hard_ranges: UniformVelocityCommandCfg.Ranges

  def build(self, env) -> CMoEVelocityCommand:
    return CMoEVelocityCommand(self, env)


__all__ = ["CMoEVelocityCommand", "CMoEVelocityCommandCfg"]
