# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE policy modules for the G1 terrain task.

The implementation keeps the five-expert mixture used by CMoE while exposing
the model protocol expected by rsl-rl 5.x.  The actor input is the current
45-dimensional proprioceptive frame, the state-estimator outputs, the 77-point
height map, and the terrain-estimator latent.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import MLP, EmpiricalNormalization, GaussianDistribution
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDictBase
from torch import Tensor


def _mlp(
  input_dim: int,
  output_dim: int,
  hidden_dims: tuple[int, ...] | list[int],
  activation: str,
  last_activation: str | None = None,
) -> MLP:
  return MLP(
    input_dim,
    output_dim,
    hidden_dims,
    activation,
    last_activation=last_activation,
  )


class StateEstimator(nn.Module):
  """VAE state estimator used by the CMoE actor."""

  def __init__(
    self,
    temporal_steps: int = 10,
    num_one_step_obs: int = 45,
    prop_enc_hidden_dims: tuple[int, ...] = (128, 64, 32),
    dec_hidden_dims: tuple[int, ...] = (32, 64, 128),
    latent_dim: int = 16,
    explicit_dim: int = 3,
    activation: str = "elu",
    learning_rate: float = 1e-3,
    max_grad_norm: float = 10.0,
    kld_weight: float = 0.005,
    use_estimation_loss: bool = True,
    use_latent_loss: bool = True,
    use_map_estimator: bool = False,
    **_: Any,
  ) -> None:
    super().__init__()
    self.temporal_steps = temporal_steps
    self.num_one_step_obs = num_one_step_obs
    self.num_latent = prop_enc_hidden_dims[-1]
    self.history_dim = temporal_steps * num_one_step_obs
    self.num_prop_obs = self.history_dim + (77 if use_map_estimator else 0)
    self.latent_dim = latent_dim
    self.explicit_dim = explicit_dim
    self.max_grad_norm = max_grad_norm
    self.kld_weight = kld_weight
    self.use_estimation_loss = use_estimation_loss
    self.use_latent_loss = use_latent_loss
    self.use_map_estimator = use_map_estimator

    self.encoder = _mlp(
      self.history_dim,
      prop_enc_hidden_dims[-1],
      prop_enc_hidden_dims[:-1],
      activation,
      last_activation=activation,
    )
    self.fc_mu = nn.Linear(prop_enc_hidden_dims[-1], latent_dim)
    self.fc_var = nn.Linear(prop_enc_hidden_dims[-1], latent_dim)
    self.fc_explicit = nn.Linear(prop_enc_hidden_dims[-1], explicit_dim)
    self.decoder = _mlp(
      latent_dim + explicit_dim,
      self.num_one_step_obs,
      dec_hidden_dims,
      activation,
    )
    self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

  def encode(self, obs_history: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    result = self.encoder(obs_history[:, : self.history_dim].detach())
    mu = self.fc_mu(result)
    log_var = self.fc_var(result)
    explicit = self.fc_explicit(result)
    z = self.reparameterize(mu, log_var) if self.training else mu
    return explicit, z, mu, log_var

  def forward(self, obs_history: Tensor) -> tuple[Tensor, Tensor]:
    explicit, z, _, _ = self.encode(obs_history)
    return explicit.detach(), z.detach()

  def get_latent(self, obs_history: Tensor) -> tuple[Tensor, Tensor]:
    explicit, z = self.forward(obs_history)
    return explicit, z

  @staticmethod
  def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    return mu + torch.rand_like(logvar) * torch.exp(0.5 * logvar)

  def update(
    self,
    obs_history: Tensor,
    critic_obs: Tensor,
    next_critic_obs: Tensor,
    lr: float | None = None,
    gradient_sync: Callable[[Iterable[nn.Parameter]], None] | None = None,
  ):
    if lr is not None:
      for group in self.optimizer.param_groups:
        group["lr"] = lr

    explicit = critic_obs[
      :, self.num_one_step_obs : self.num_one_step_obs + self.explicit_dim
    ].detach()
    next_obs = next_critic_obs[:, 3 : self.num_one_step_obs + 3].detach()
    pred_explicit, z, mu, log_var = self.encode(obs_history)
    pred_next_obs = self.decoder(torch.cat((z, pred_explicit), dim=-1))
    recons_loss = F.mse_loss(pred_next_obs, next_obs)
    kld_loss = torch.mean(
      -0.5 * torch.sum(1 + log_var - mu.square() - log_var.exp(), dim=1), dim=0
    )
    vae_loss = recons_loss + self.kld_weight * kld_loss
    estimation_loss = F.mse_loss(pred_explicit, explicit)
    losses = (
      self.use_estimation_loss * estimation_loss + self.use_latent_loss * vae_loss
    )
    self.optimizer.zero_grad()
    losses.backward()
    if gradient_sync is not None:
      gradient_sync(self.parameters())
    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    self.optimizer.step()
    return (
      estimation_loss.item(),
      vae_loss.item(),
      recons_loss.item(),
      kld_loss.item(),
    )


class TerrainEstimator(nn.Module):
  """Height-map encoder used by the CMoE actor."""

  def __init__(
    self,
    temporal_steps: int = 10,
    num_one_step_obs: int = 45,
    terrain_dim: int = 77,
    prop_enc_hidden_dims: tuple[int, ...] = (128, 64, 32),
    dec_hidden_dims: tuple[int, ...] = (32, 64, 128),
    latent_dim: int = 16,
    activation: str = "elu",
    learning_rate: float = 1e-3,
    max_grad_norm: float = 10.0,
    use_latent_loss: bool = True,
    use_map_estimator: bool = False,
    **_: Any,
  ) -> None:
    super().__init__()
    self.temporal_steps = temporal_steps
    self.num_one_step_obs = num_one_step_obs
    self.num_latent = prop_enc_hidden_dims[-1]
    self.history_dim = temporal_steps * num_one_step_obs
    self.num_prop_obs = self.history_dim
    self.terrain_dim = terrain_dim
    self.latent_dim = latent_dim
    self.max_grad_norm = max_grad_norm
    self.kld_weight = 0.005
    self.use_estimation_loss = True
    self.use_latent_loss = use_latent_loss
    self.use_map_estimator = use_map_estimator
    self.encoder = _mlp(
      terrain_dim,
      prop_enc_hidden_dims[-1],
      prop_enc_hidden_dims[:-1],
      activation,
      last_activation=activation,
    )
    self.fc_mu = nn.Linear(prop_enc_hidden_dims[-1], latent_dim)
    self.fc_var = nn.Linear(prop_enc_hidden_dims[-1], latent_dim)
    self.decoder = _mlp(latent_dim, terrain_dim, dec_hidden_dims, activation)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

  def encode(self, obs_history: Tensor) -> Tensor:
    terrain = obs_history[:, self.history_dim : self.history_dim + self.terrain_dim]
    return self.fc_mu(self.encoder(terrain.detach()))

  def forward(self, obs_history: Tensor) -> Tensor:
    return self.encode(obs_history).detach()

  def get_latent(self, obs_history: Tensor) -> Tensor:
    return self.forward(obs_history)

  def update(
    self,
    obs_history: Tensor,
    next_critic_obs: Tensor,
    lr: float | None = None,
    gradient_sync: Callable[[Iterable[nn.Parameter]], None] | None = None,
  ):
    if lr is not None:
      for group in self.optimizer.param_groups:
        group["lr"] = lr
    terrain = obs_history[
      :, self.history_dim : self.history_dim + self.terrain_dim
    ].detach()
    latent = self.encode(obs_history)
    pred_terrain = self.decoder(latent)
    recons_loss = F.mse_loss(pred_terrain, terrain)
    loss = self.use_latent_loss * recons_loss
    self.optimizer.zero_grad()
    loss.backward()
    if gradient_sync is not None:
      gradient_sync(self.parameters())
    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    self.optimizer.step()
    return 0.0, recons_loss.item(), 0.0, 0.0


class ExpertActorCritic(nn.Module):
  """One CMoE expert, retaining both actor and critic heads."""

  def __init__(
    self,
    actor_input_dim: int,
    critic_input_dim: int,
    num_actions: int,
    actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.std = nn.Parameter(torch.ones(num_actions))
    self.actor = _mlp(actor_input_dim, num_actions, actor_hidden_dims, activation)
    self.critic = _mlp(critic_input_dim, 1, critic_hidden_dims, activation)

  def forward(self, actor_input: Tensor) -> Tensor:
    return self.actor(actor_input)

  def act(self, actor_input: Tensor) -> tuple[Tensor, Tensor]:
    return self.actor(actor_input), self.std

  def act_inference(self, actor_input: Tensor) -> Tensor:
    return self.actor(actor_input)

  def evaluate(self, critic_input: Tensor) -> Tensor:
    return self.critic(critic_input)


class CMoEModel(nn.Module):
  """Five-expert CMoE actor implementing the rsl-rl model protocol."""

  is_recurrent = False

  def __init__(
    self,
    obs: TensorDictBase,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
    num_one_step_obs: int = 45,
    temporal_steps: int = 10,
    terrain_dim: int = 77,
    state_latent_dim: int = 16,
    terrain_latent_dim: int = 16,
    explicit_dim: int = 3,
    num_experts: int = 5,
    num_prototypes: int = 32,
    temperature: float = 0.2,
    estimator_hidden_dims: tuple[int, ...] = (128, 64, 32),
    estimator_decoder_dims: tuple[int, ...] = (32, 64, 128),
    critic_hidden_dims: tuple[int, ...] | list[int] | None = None,
    use_detailed_explicit: bool = False,
    use_estimation_loss: bool = True,
    use_latent_loss: bool = True,
    use_map_estimator: bool = False,
    **_: Any,
  ) -> None:
    super().__init__()
    self.obs_groups = obs_groups[obs_set]
    self.obs_dim = sum(obs[group].shape[-1] for group in self.obs_groups)
    self.critic_groups = obs_groups.get("critic", self.obs_groups)
    self.critic_obs_dim = sum(obs[group].shape[-1] for group in self.critic_groups)
    self.obs_normalization = obs_normalization
    self.critic_obs_normalization = False
    self.obs_normalizer = (
      EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
    )
    self.critic_obs_normalizer = nn.Identity()

    self.num_one_step_obs = num_one_step_obs
    self.temporal_steps = temporal_steps
    self.history_dim = temporal_steps * num_one_step_obs
    self.terrain_dim = terrain_dim
    self.state_latent_dim = state_latent_dim
    self.terrain_latent_dim = terrain_latent_dim
    self.explicit_dim = explicit_dim + (11 if use_detailed_explicit else 0)
    self.num_experts = num_experts
    self.num_actions = output_dim
    self.temperature = temperature

    self.state_estimator = StateEstimator(
      temporal_steps=temporal_steps,
      num_one_step_obs=num_one_step_obs,
      prop_enc_hidden_dims=estimator_hidden_dims,
      dec_hidden_dims=estimator_decoder_dims,
      latent_dim=state_latent_dim,
      explicit_dim=self.explicit_dim,
      activation=activation,
      use_estimation_loss=use_estimation_loss,
      use_latent_loss=use_latent_loss,
      use_map_estimator=use_map_estimator,
    )
    self.terrain_estimator = TerrainEstimator(
      temporal_steps=temporal_steps,
      num_one_step_obs=num_one_step_obs,
      terrain_dim=terrain_dim,
      prop_enc_hidden_dims=estimator_hidden_dims,
      dec_hidden_dims=estimator_decoder_dims,
      latent_dim=terrain_latent_dim,
      activation=activation,
      use_latent_loss=use_latent_loss,
      use_map_estimator=use_map_estimator,
    )

    self.actor_input_dim = (
      num_one_step_obs
      + self.explicit_dim
      + state_latent_dim
      + terrain_dim
      + terrain_latent_dim
    )
    critic_hidden_dims = critic_hidden_dims or tuple(hidden_dims)
    self.experts = nn.ModuleList(
      [
        ExpertActorCritic(
          self.actor_input_dim,
          self.critic_obs_dim,
          output_dim,
          hidden_dims,
          critic_hidden_dims,
          activation,
        )
        for _ in range(num_experts)
      ]
    )
    self.gating_network = nn.Sequential(
      nn.Linear(self.actor_input_dim, 128),
      resolve_nn_activation(activation),
      nn.Linear(128, num_experts),
      nn.Softmax(dim=-1),
    )
    self.gate_projector = nn.Sequential(
      nn.Linear(num_experts, 128),
      resolve_nn_activation(activation),
      nn.Linear(128, 64),
      resolve_nn_activation(activation),
      nn.Linear(64, 16),
    )
    self.terrain_projector = nn.Sequential(
      nn.Linear(terrain_dim + terrain_latent_dim, 128),
      resolve_nn_activation(activation),
      nn.Linear(128, 64),
      resolve_nn_activation(activation),
      nn.Linear(64, 16),
    )
    self.prototypes = nn.Embedding(num_prototypes, 16)

    cfg = dict(distribution_cfg or {})
    cfg.pop("class_name", None)
    self.distribution = GaussianDistribution(output_dim, **cfg)
    self.gate_weights = torch.empty(0)

  def _obs_tensor(self, obs: TensorDictBase | Tensor, normalize: bool = True) -> Tensor:
    if isinstance(obs, TensorDictBase):
      tensor = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    else:
      tensor = obs
    return self.obs_normalizer(tensor) if normalize else tensor

  def _critic_tensor(self, obs: TensorDictBase | Tensor) -> Tensor:
    if isinstance(obs, TensorDictBase):
      tensor = torch.cat([obs[group] for group in self.critic_groups], dim=-1)
    else:
      tensor = obs
    return self.critic_obs_normalizer(tensor)

  def get_latent(
    self,
    obs: TensorDictBase | Tensor,
    masks: Tensor | None = None,
    hidden_state: Any = None,
  ) -> Tensor:
    del masks, hidden_state
    return self._obs_tensor(obs)

  def _actor_input(self, obs_history: Tensor) -> Tensor:
    explicit, latent = self.state_estimator(obs_history)
    terrain = obs_history[:, -self.terrain_dim :]
    terrain_latent = self.terrain_estimator(obs_history)
    return torch.cat(
      (
        obs_history[:, : self.num_one_step_obs],
        explicit,
        latent,
        terrain,
        terrain_latent,
      ),
      dim=-1,
    )

  def _actor_mean(self, obs_history: Tensor) -> tuple[Tensor, Tensor]:
    actor_input = self._actor_input(obs_history)
    gate_weights = self.gating_network(actor_input)
    expert_means = torch.stack(
      [expert.act(actor_input)[0] for expert in self.experts], dim=1
    )
    mean = (expert_means * gate_weights.unsqueeze(-1)).sum(dim=1)
    return mean, gate_weights

  def forward(
    self,
    obs: TensorDictBase | Tensor,
    masks: Tensor | None = None,
    hidden_state: Any = None,
    stochastic_output: bool = False,
  ) -> Tensor:
    del masks, hidden_state
    mean, self.gate_weights = self._actor_mean(self._obs_tensor(obs))
    if stochastic_output:
      self.distribution.update(mean)
      return self.distribution.sample()
    return mean

  def act(self, obs: TensorDictBase | Tensor, **kwargs: Any) -> Tensor:
    return self.forward(obs, stochastic_output=True, **kwargs)

  def act_inference(
    self, obs: TensorDictBase | Tensor, observations: Any = None
  ) -> Tensor:
    del observations
    return self.forward(obs)

  def evaluate(
    self, critic_observations: TensorDictBase | Tensor, **kwargs: Any
  ) -> Tensor:
    gate_weights = kwargs.pop("gate_weights", self.gate_weights).detach()
    critic_input = self._critic_tensor(critic_observations)
    values = torch.stack(
      [expert.evaluate(critic_input) for expert in self.experts], dim=1
    )
    return (values * gate_weights.unsqueeze(-1)).sum(dim=1)

  def update_estimators(
    self,
    obs_batch: TensorDictBase | Tensor,
    critic_obs_batch: TensorDictBase | Tensor,
    next_critic_obs_batch: TensorDictBase | Tensor,
    lr: float | None = None,
    gradient_sync: Callable[[Iterable[nn.Parameter]], None] | None = None,
  ):
    obs = self._obs_tensor(obs_batch)
    critic_obs = self._critic_tensor(critic_obs_batch)
    next_critic_obs = self._critic_tensor(next_critic_obs_batch)
    state_loss = self.state_estimator.update(
      obs, critic_obs, next_critic_obs, lr, gradient_sync
    )
    terrain_loss = self.terrain_estimator.update(
      obs, next_critic_obs, lr, gradient_sync
    )
    return (*state_loss, *terrain_loss)

  def compute_contrastive_loss(self, obs: TensorDictBase | Tensor, **_: Any) -> Tensor:
    obs_history = self._obs_tensor(obs).detach()
    with torch.no_grad():
      latent2 = self.terrain_estimator(obs_history)
    actor_input = self._actor_input(obs_history).detach()
    gate_weights = self.gating_network(actor_input)
    terrain_input = torch.cat((obs_history[:, -self.terrain_dim :], latent2), dim=-1)
    gate_z = F.normalize(self.gate_projector(gate_weights), dim=-1)
    terrain_z = F.normalize(self.terrain_projector(terrain_input), dim=-1)
    with torch.no_grad():
      self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=-1))
      q_s = sinkhorn(gate_z @ self.prototypes.weight.T)
      q_t = sinkhorn(terrain_z @ self.prototypes.weight.T)
    score_s = gate_z @ self.prototypes.weight.T
    score_t = terrain_z @ self.prototypes.weight.T
    log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
    log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)
    return -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

  def reset(self, dones: Tensor | None = None, hidden_state: Any = None) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> None:
    return None

  def detach_hidden_state(self, dones: Tensor | None = None) -> None:
    del dones

  def update_normalization(self, obs: TensorDictBase) -> None:
    if self.obs_normalization:
      self.obs_normalizer.update(self._obs_tensor(obs, normalize=False))

  @property
  def output_mean(self) -> Tensor:
    return self.distribution.mean

  @property
  def output_std(self) -> Tensor:
    return self.distribution.std

  @property
  def output_entropy(self) -> Tensor:
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[Tensor, ...]:
    return self.distribution.params

  def get_output_log_prob(self, outputs: Tensor) -> Tensor:
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self,
    old_params: tuple[Tensor, ...],
    new_params: tuple[Tensor, ...],
  ) -> Tensor:
    return self.distribution.kl_divergence(old_params, new_params)

  @property
  def action_mean(self) -> Tensor:
    return self.output_mean

  @property
  def action_std(self) -> Tensor:
    return self.output_std

  @property
  def entropy(self) -> Tensor:
    return self.output_entropy

  def get_actions_log_prob(self, actions: Tensor) -> Tensor:
    return self.get_output_log_prob(actions)

  def as_jit(self) -> nn.Module:
    return _CMoEExport(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _CMoEExport(self, verbose=verbose)


class _CMoEExport(nn.Module):
  def __init__(self, model: CMoEModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.state_estimator = copy.deepcopy(model.state_estimator)
    self.terrain_estimator = copy.deepcopy(model.terrain_estimator)
    self.experts = copy.deepcopy(model.experts)
    self.gating_network = copy.deepcopy(model.gating_network)
    self.num_one_step_obs = model.num_one_step_obs
    self.terrain_dim = model.terrain_dim
    self.input_size = model.obs_dim

  def forward(self, obs: Tensor) -> Tensor:
    obs = self.obs_normalizer(obs)
    explicit, latent = self.state_estimator(obs)
    terrain = obs[:, -self.terrain_dim :]
    terrain_latent = self.terrain_estimator(obs)
    actor_input = torch.cat(
      (obs[:, : self.num_one_step_obs], explicit, latent, terrain, terrain_latent),
      dim=-1,
    )
    gate_weights = self.gating_network(actor_input)
    expert_means = torch.stack(
      [expert.actor(actor_input) for expert in self.experts], dim=1
    )
    return (expert_means * gate_weights.unsqueeze(-1)).sum(dim=1)

  @torch.jit.export
  def reset(self) -> None:
    pass

  def get_dummy_inputs(self) -> tuple[Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


@torch.no_grad()
def sinkhorn(out: Tensor, eps: float = 0.05, iters: int = 3) -> Tensor:
  q = torch.exp(out / eps).T
  num_prototypes, batch_size = q.shape
  q /= q.sum()
  for _ in range(iters):
    q /= q.sum(dim=1, keepdim=True)
    q /= num_prototypes
    q /= q.sum(dim=0, keepdim=True)
    q /= batch_size
  return (q * batch_size).T


__all__ = [
  "CMoEModel",
  "ExpertActorCritic",
  "StateEstimator",
  "TerrainEstimator",
  "sinkhorn",
]
