"""SMP steering tasks — registers ``Smp-Steering-CBot`` and ``Smp-Forward-CBot``
on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import cbot_smp_ppo_runner_cfg
from smp.rl.tasks.config.cbotv3_2.forward_env_cfg import cbot_forward_smp_env_cfg
from smp.rl.tasks.config.cbotv3_2.steering_env_cfg import cbot_steering_smp_env_cfg

_steering_rl = cbot_smp_ppo_runner_cfg()
_steering_rl.experiment_name = "smp_steering_cbot"
_steering_rl.run_name = "smp_steering_cbot"

register_mjlab_task(
  task_id="Smp-Steering-CBot",
  env_cfg=cbot_steering_smp_env_cfg(play=False),
  play_env_cfg=cbot_steering_smp_env_cfg(play=True),
  rl_cfg=_steering_rl,
)

_forward_rl = cbot_smp_ppo_runner_cfg()
_forward_rl.experiment_name = "smp_forward_cbot"
_forward_rl.run_name = "smp_forward_cbot"

register_mjlab_task(
  task_id="Smp-Forward-CBot",
  env_cfg=cbot_forward_smp_env_cfg(play=False),
  play_env_cfg=cbot_forward_smp_env_cfg(play=True),
  rl_cfg=_forward_rl,
)

__all__ = [
  "cbot_forward_smp_env_cfg",
  "cbot_steering_smp_env_cfg",
]
