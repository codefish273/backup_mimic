# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).
# Adapted from rsl_rl (BSD-3-Clause, Copyright (c) 2021 ETH Zurich, Nikita Rudin
# and NVIDIA CORPORATION & AFFILIATES). See rsl_rl/LICENSE.

"""Rollout storage used by the original CMoE PPO loop.

The critic observation at the next transition is kept separately from the
observation fed to the next policy step.  This is required when an environment
is reset in-place after a termination: the estimator target must remain the
terminal privileged observation.
"""

from __future__ import annotations

from typing import Iterator

import torch
from tensordict import TensorDict


class RolloutStorage:
  class Transition:
    def __init__(self) -> None:
      self.observations: TensorDict | None = None
      self.critic_observations: TensorDict | None = None
      self.next_critic_observations: TensorDict | None = None
      self.actions: torch.Tensor | None = None
      self.rewards: torch.Tensor | None = None
      self.dones: torch.Tensor | None = None
      self.values: torch.Tensor | None = None
      self.actions_log_prob: torch.Tensor | None = None
      self.action_mean: torch.Tensor | None = None
      self.action_sigma: torch.Tensor | None = None

    def clear(self) -> None:
      self.__init__()

  class Batch:
    def __init__(
      self,
      observations: TensorDict,
      critic_observations: TensorDict,
      next_critic_observations: TensorDict,
      actions: torch.Tensor,
      target_values: torch.Tensor,
      advantages: torch.Tensor,
      returns: torch.Tensor,
      old_actions_log_prob: torch.Tensor,
      old_mu: torch.Tensor,
      old_sigma: torch.Tensor,
    ) -> None:
      self.observations = observations
      self.critic_observations = critic_observations
      self.next_critic_observations = next_critic_observations
      self.actions = actions
      self.target_values = target_values
      self.advantages = advantages
      self.returns = returns
      self.old_actions_log_prob = old_actions_log_prob
      self.old_mu = old_mu
      self.old_sigma = old_sigma

  def __init__(
    self,
    num_envs: int,
    num_transitions_per_env: int,
    observations: TensorDict,
    critic_observations: TensorDict,
    actions_shape: tuple[int, ...] | list[int],
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.num_envs = num_envs
    self.num_transitions_per_env = num_transitions_per_env
    self.actions_shape = actions_shape

    self.observations = self._zeros(observations)
    self.critic_observations = self._zeros(critic_observations)
    self.next_critic_observations = self._zeros(critic_observations)
    self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.actions = torch.zeros(
      num_transitions_per_env, num_envs, *actions_shape, device=device
    )
    self.dones = torch.zeros(
      num_transitions_per_env, num_envs, 1, device=device, dtype=torch.uint8
    )
    self.actions_log_prob = torch.zeros(
      num_transitions_per_env, num_envs, 1, device=device
    )
    self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
    self.advantages = torch.zeros(
      num_transitions_per_env, num_envs, 1, device=device
    )
    self.mu = torch.zeros(
      num_transitions_per_env, num_envs, *actions_shape, device=device
    )
    self.sigma = torch.zeros(
      num_transitions_per_env, num_envs, *actions_shape, device=device
    )
    self.step = 0

  def _zeros(self, source: TensorDict) -> TensorDict:
    return TensorDict(
      {
        key: torch.zeros(
          self.num_transitions_per_env,
          *value.shape,
          dtype=value.dtype,
          device=self.device,
        )
        for key, value in source.items()
      },
      batch_size=[self.num_transitions_per_env, self.num_envs],
      device=self.device,
    )

  def add_transitions(self, transition: Transition) -> None:
    if self.step >= self.num_transitions_per_env:
      raise AssertionError("Rollout buffer overflow")
    self.observations[self.step].copy_(transition.observations)
    self.critic_observations[self.step].copy_(transition.critic_observations)
    self.next_critic_observations[self.step].copy_(
      transition.next_critic_observations
    )
    self.actions[self.step].copy_(transition.actions)
    self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.values[self.step].copy_(transition.values)
    self.actions_log_prob[self.step].copy_(
      transition.actions_log_prob.view(-1, 1)
    )
    self.mu[self.step].copy_(transition.action_mean)
    self.sigma[self.step].copy_(transition.action_sigma)
    self.step += 1

  def clear(self) -> None:
    self.step = 0

  def get_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
    done = self.dones
    done[-1] = 1
    flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
    done_indices = torch.cat(
      (
        flat_dones.new_tensor([-1], dtype=torch.int64),
        flat_dones.nonzero(as_tuple=False)[:, 0],
      )
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    return trajectory_lengths.float().mean(), self.rewards.mean()

  def compute_returns(
    self, last_values: torch.Tensor, gamma: float, lam: float
  ) -> None:
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):
      next_values = (
        last_values
        if step == self.num_transitions_per_env - 1
        else self.values[step + 1]
      )
      next_is_not_terminal = 1.0 - self.dones[step].float()
      delta = (
        self.rewards[step]
        + next_is_not_terminal * gamma * next_values
        - self.values[step]
      )
      advantage = delta + next_is_not_terminal * gamma * lam * advantage
      self.returns[step] = advantage + self.values[step]

    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()) / (
      self.advantages.std() + 1e-8
    )

  def mini_batch_generator(
    self, num_mini_batches: int, num_epochs: int = 8
  ) -> Iterator[Batch]:
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(
      num_mini_batches * mini_batch_size,
      requires_grad=False,
      device=self.device,
    )

    observations = self.observations.flatten(0, 1)
    critic_observations = self.critic_observations.flatten(0, 1)
    next_critic_observations = self.next_critic_observations.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    target_values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_mu = self.mu.flatten(0, 1)
    old_sigma = self.sigma.flatten(0, 1)

    for _ in range(num_epochs):
      for index in range(num_mini_batches):
        batch_idx = indices[
          index * mini_batch_size : (index + 1) * mini_batch_size
        ]
        yield self.Batch(
          observations=observations[batch_idx],
          critic_observations=critic_observations[batch_idx],
          next_critic_observations=next_critic_observations[batch_idx],
          actions=actions[batch_idx],
          target_values=target_values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_actions_log_prob[batch_idx],
          old_mu=old_mu[batch_idx],
          old_sigma=old_sigma[batch_idx],
        )


__all__ = ["RolloutStorage"]
