"""Optional native-viewer arrow-key control for SMP steering tasks."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import mjlab.scripts.play as mjlab_play
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native import NativeMujocoViewer
from mjlab.viewer.native.keys import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP


@dataclass(frozen=True)
class KeyboardPlayConfig(mjlab_play.PlayConfig):
  """Extend mjlab's play CLI with SMP keyboard steering."""

  keyboard_control: bool = False
  """Use arrow keys to increment heading-frame velocity by 0.2 m/s."""


class ArrowKeySteeringViewer(NativeMujocoViewer):
  """Native viewer that safely applies arrow-key events on the main thread."""

  _SPEED_STEP = 0.2
  _MAX_FORWARD_SPEED = 5.0

  def __init__(self, env: Any, policy: Any, *args: Any, **kwargs: Any) -> None:
    super().__init__(env, policy, *args, **kwargs)
    if env.num_envs != 1:
      raise ValueError(
        f"--keyboard-control requires --num-envs 1, got {env.num_envs}"
      )
    try:
      self._steering_command = env.unwrapped.command_manager.get_term("steering")
    except KeyError as exc:
      raise ValueError(
        "--keyboard-control requires a task with a 'steering' command"
      ) from exc
    if not hasattr(self._steering_command, "enable_keyboard_control"):
      raise ValueError(
        "This task's 'steering' command does not support --keyboard-control"
      )
    self._steering_command.enable_keyboard_control()
    print(
      "[INFO] Keyboard control enabled: "
      "↑/↓ adjust forward speed, ←/→ adjust lateral speed "
      "(0.2 m/s per press, forward limit 5.0 m/s)"
    )

  def _safe_key_callback(self, key: int) -> None:
    """Queue arrow keys without touching environment tensors on the viewer thread."""
    if key in (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT):
      self.request_action("KEYBOARD_STEERING", key)
      return
    super()._safe_key_callback(key)

  def _handle_custom_action(
    self, action: ViewerAction, payload: object | None
  ) -> bool:
    if action == ViewerAction.CUSTOM and payload in (
      KEY_UP,
      KEY_DOWN,
      KEY_LEFT,
      KEY_RIGHT,
    ):
      forward_delta = self._SPEED_STEP * (
        int(payload == KEY_UP) - int(payload == KEY_DOWN)
      )
      left_delta = self._SPEED_STEP * (
        int(payload == KEY_LEFT) - int(payload == KEY_RIGHT)
      )
      forward, left = self._steering_command.adjust_keyboard_velocity(
        self.env_idx,
        forward_delta,
        left_delta,
        forward_speed_max=self._MAX_FORWARD_SPEED,
      )
      key_name = {
        KEY_UP: "↑",
        KEY_DOWN: "↓",
        KEY_LEFT: "←",
        KEY_RIGHT: "→",
      }[payload]
      print(
        f"[INFO] Key {key_name}: forward={forward:+.1f} m/s, "
        f"left={left:+.1f} m/s, speed={math.hypot(forward, left):.1f} m/s",
        flush=True,
      )
      return True
    return super()._handle_custom_action(action, payload)


def main() -> None:
  """Run mjlab play with the optional keyboard-control extension installed."""
  original_run_play = mjlab_play.run_play

  def run_play_with_keyboard(task_id: str, cfg: KeyboardPlayConfig) -> None:
    if cfg.keyboard_control:
      if cfg.num_envs is not None and cfg.num_envs != 1:
        raise ValueError(
          f"--keyboard-control requires --num-envs 1, got {cfg.num_envs}"
        )
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      if cfg.viewer == "viser" or (cfg.viewer == "auto" and not has_display):
        raise ValueError("--keyboard-control requires the native viewer")
      original_viewer = mjlab_play.NativeMujocoViewer
      mjlab_play.NativeMujocoViewer = ArrowKeySteeringViewer
      try:
        original_run_play(task_id, cfg)
      finally:
        mjlab_play.NativeMujocoViewer = original_viewer
    else:
      original_run_play(task_id, cfg)

  mjlab_play.PlayConfig = KeyboardPlayConfig
  mjlab_play.run_play = run_play_with_keyboard
  mjlab_play.main()
