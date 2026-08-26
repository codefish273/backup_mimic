# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""RSL-RL configuration for CMoE-G1."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class CMoEModelCfg(RslRlModelCfg):
  class_name: str = "smp.rl.cmoe.model:CMoEModel"
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  distribution_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "init_std": 1.0,
    }
  )
  num_one_step_obs: int = 45
  temporal_steps: int = 10
  terrain_dim: int = 77
  state_latent_dim: int = 16
  terrain_latent_dim: int = 16
  explicit_dim: int = 3
  num_experts: int = 5
  num_prototypes: int = 32
  temperature: float = 0.2
  estimator_hidden_dims: tuple[int, ...] = (128, 64, 32)
  estimator_decoder_dims: tuple[int, ...] = (32, 64, 128)
  use_detailed_explicit: bool = False
  use_estimation_loss: bool = True
  use_latent_loss: bool = True
  use_map_estimator: bool = True


@dataclass
class CMoEPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "smp.rl.cmoe.algorithm:CMoEPPO"
  rnd_cfg: dict[str, Any] | None = None
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  entropy_coef: float = 0.01
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1e-3
  schedule: str = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  optimizer: str = "adam"


def g1_cmoe_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=1,
    actor=CMoEModelCfg(),
    critic=RslRlModelCfg(),
    algorithm=CMoEPpoAlgorithmCfg(),
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    experiment_name="g1_cmoe",
    run_name="cmoe",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=50_000,
    clip_actions=100.0,
    logger="tensorboard",
    upload_model=False,
  )


__all__ = [
  "CMoEModelCfg",
  "CMoEPpoAlgorithmCfg",
  "g1_cmoe_ppo_runner_cfg",
]
