"""Run with: .venv/bin/python -m unittest discover -s tests -v."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from scripts.csv_to_npz import (
  JOINT_NAMES,
  RobotMotionLoader,
  _compute_windows,
  _read_csv_motion,
  _robot_config,
)

from smp.rl.assets.cbotv3.cbot_v3 import (
  CBOT_V3_NOHEAD_EE_BODY_NAMES,
  CBOT_V3_NOHEAD_JOINT_NAMES,
)
from smp.rl.utils import MotionFeatureBuffer


class CsvMotionTests(unittest.TestCase):
  def setUp(self) -> None:
    self.temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp_dir.cleanup)
    self.path = Path(self.temp_dir.name) / "motion.csv"
    self.joints = CBOT_V3_NOHEAD_JOINT_NAMES
    self.values = np.zeros((6, 7 + len(self.joints)))
    self.values[:, 0] = np.arange(6) * 0.02
    self.values[:, 2] = 1.0
    self.values[:, 5] = np.sin(np.arange(6) * 0.01)
    self.values[:, 6] = np.cos(np.arange(6) * 0.01)
    self.values[:, 7:] = np.arange(len(self.joints)) * 0.01
    self.header = [
      "root_x",
      "root_y",
      "root_z",
      "root_qx",
      "root_qy",
      "root_qz",
      "root_qw",
      *(f"dof_{name}" for name in self.joints),
    ]

  def write_named(
    self,
    metadata: str = "# sample_rate: 50\n",
    with_time: bool = True,
    header: list[str] | None = None,
    values: np.ndarray | None = None,
  ) -> None:
    header = self.header if header is None else header
    values = self.values if values is None else values
    # Reverse all columns so positional parsing cannot accidentally pass.
    header = list(reversed(header))
    values = values[:, ::-1]
    if with_time:
      header = ["time", *header]
      values = np.column_stack((np.arange(values.shape[0]) / 50.0, values))
    with self.path.open("w", newline="") as dest:
      dest.write(metadata)
      writer = csv.writer(dest)
      writer.writerow(header)
      writer.writerows(values)

  def test_named_cbot_reorders_joints_and_infers_fps(self) -> None:
    self.write_named()
    values, fps = _read_csv_motion(self.path, self.joints, None)
    np.testing.assert_array_equal(values, self.values)
    self.assertEqual(fps, 50.0)
    motion = RobotMotionLoader(self.path, self.joints, None, 50, "cpu")
    self.assertEqual(motion.output_frames, 5)
    torch.testing.assert_close(
      motion.motion_dof_poss[0], torch.tensor(self.values[0, 7:], dtype=torch.float32)
    )
    torch.testing.assert_close(
      motion.motion_base_lin_vels[:, 0], torch.ones(motion.output_frames)
    )

  def test_fps_sources_and_explicit_override(self) -> None:
    for metadata, with_time, override, expected in (
      ("# sample_rate: 50\n", False, None, 50),
      ("", True, None, 50),
      ("", False, None, 30),
      ("# sample_rate: 50\n", True, 60, 60),
    ):
      with self.subTest(metadata=metadata, with_time=with_time, override=override):
        self.write_named(metadata, with_time)
        _, fps = _read_csv_motion(self.path, self.joints, override)
        self.assertEqual(fps, expected)

  def test_g1_matches_original_loader(self) -> None:
    np.savetxt(self.path, self.values, delimiter=",")
    old = CsvMotionLoader(str(self.path), 30, 50, "cpu")
    new = RobotMotionLoader(self.path, JOINT_NAMES, None, 50, "cpu")
    for field in (
      "motion_base_poss",
      "motion_base_rots",
      "motion_dof_poss",
      "motion_base_lin_vels",
      "motion_base_ang_vels",
      "motion_dof_vels",
    ):
      with self.subTest(field=field):
        torch.testing.assert_close(
          getattr(old, field), getattr(new, field), rtol=0, atol=0
        )

  def test_rejects_wrong_joint_columns(self) -> None:
    self.write_named()
    with self.assertRaisesRegex(ValueError, "selected robot"):
      _read_csv_motion(self.path, JOINT_NAMES, None)
    self.write_named(header=self.header[:-1], values=self.values[:, :-1])
    with self.assertRaisesRegex(ValueError, "missing=.*right_wrist_yaw_joint"):
      _read_csv_motion(self.path, self.joints, None)
    np.savetxt(self.path, self.values[:, :-1], delimiter=",")
    with self.assertRaisesRegex(ValueError, "expected 36 columns"):
      _read_csv_motion(self.path, self.joints, None)

  def test_rejects_duplicate_headers_and_extra_joints(self) -> None:
    header = [*self.header[:-1], self.header[0]]
    self.write_named(header=header)
    with self.assertRaisesRegex(ValueError, "duplicate"):
      _read_csv_motion(self.path, self.joints, None)
    self.write_named(
      header=[*self.header, "dof_head_yaw_joint"],
      values=np.column_stack((self.values, np.zeros(6))),
    )
    with self.assertRaisesRegex(ValueError, "unexpected=.*head_yaw_joint"):
      _read_csv_motion(self.path, self.joints, None)

  def test_rejects_invalid_numeric_data(self) -> None:
    cases = [
      (self.values[:1], "at least two"),
      (np.full_like(self.values, np.nan), "finite"),
      (np.zeros_like(self.values), "quaternion"),
    ]
    for values, message in cases:
      with self.subTest(message=message):
        np.savetxt(self.path, values, delimiter=",")
        with self.assertRaisesRegex(ValueError, message):
          _read_csv_motion(self.path, self.joints, None)

  def test_rejects_invalid_time_and_fps(self) -> None:
    for times in ([0, 0.02, 0.04, 0.04, 0.08, 0.1], [0, 0.02, 0.07, 0.08, 0.1, 0.12]):
      self.write_named(
        with_time=False,
        header=["time", *self.header],
        values=np.column_stack((times, self.values)),
      )
      with self.assertRaisesRegex(ValueError, "uniformly spaced"):
        _read_csv_motion(self.path, self.joints, None)
    self.write_named(metadata="# sample_rate: 30\n")
    with self.assertRaisesRegex(ValueError, "disagrees"):
      _read_csv_motion(self.path, self.joints, None)
    self.write_named()
    for fps in (0, -1, float("nan"), float("inf")):
      with (
        self.subTest(fps=fps),
        self.assertRaisesRegex(ValueError, "positive and finite"),
      ):
        _read_csv_motion(self.path, self.joints, fps)

  def test_short_resampled_motion_has_actionable_error(self) -> None:
    self.write_named()
    with self.assertRaisesRegex(ValueError, "three resampled frames"):
      RobotMotionLoader(self.path, self.joints, None, 10, "cpu")

  def test_cbot_model_resolves_training_feature_order(self) -> None:
    scene_cfg, joint_names, ee_names = _robot_config("cbot")
    self.assertEqual(joint_names, CBOT_V3_NOHEAD_JOINT_NAMES)
    self.assertEqual(ee_names, CBOT_V3_NOHEAD_EE_BODY_NAMES)
    scene = Scene(scene_cfg, device="cpu")
    model = scene.compile()
    robot = scene["robot"]
    self.assertEqual(
      tuple(robot.find_joints(joint_names, preserve_order=True)[1]), joint_names
    )
    self.assertEqual(
      tuple(robot.find_bodies(ee_names, preserve_order=True)[1]), ee_names
    )
    self.assertEqual(model.nq, 7 + 29)
    self.assertEqual(model.body(1).name, "robot/pelvis")

  def test_windows_match_online_rl_buffer(self) -> None:
    self.write_named()
    motion = RobotMotionLoader(self.path, self.joints, None, 50, "cpu")
    n = motion.output_frames
    ee_pos = motion.motion_base_poss[:, None] + torch.arange(15).reshape(1, 5, 3) * 0.1
    states = (
      motion.motion_base_poss,
      motion.motion_base_rots,
      motion.motion_base_lin_vels,
      motion.motion_base_ang_vels,
      ee_pos,
      motion.motion_dof_poss,
    )
    windows = _compute_windows(*states, window_size=3, stride=2)
    self.assertIsNotNone(windows)
    assert windows is not None
    self.assertEqual(tuple(windows.shape), (2, 3, 59))
    buffer = MotionFeatureBuffer(1, 3, 29, 5, "cpu")
    for window_index, start in enumerate(range(0, n - 2, 2)):
      buffer.reset(
        torch.tensor([0]),
        *(state[None, start : start + 3] for state in states),
        motion.motion_dof_vels[None, start : start + 3],
      )
      torch.testing.assert_close(buffer.compute_features()[0], windows[window_index])


if __name__ == "__main__":
  unittest.main()
