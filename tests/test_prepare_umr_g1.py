"""Focused checks for the independent UMR G1 data preparation path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts.csv_to_npz import JOINT_NAMES
from scripts.prepare_umr_g1 import digest, read_result


class ResultTests(unittest.TestCase):
  def setUp(self) -> None:
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    bvh = self.root / "run.bvh"
    bvh.write_text("fixture")
    self.source = {
      "motion": "run",
      "frames": 12,
      "sha256": digest(bvh),
      "bvh": str(bvh),
    }
    human = self.root / "human.npz"
    np.savez(human, source_sha256=self.source["sha256"], poses=np.zeros((12, 55, 3)))
    qpos = np.zeros((12, 36))
    qpos[:, 3] = 1
    qpos[:, 7:] = np.arange(29)
    self.payload = {
      "qpos": qpos,
      "robot_joint_names": np.array(JOINT_NAMES[::-1]),
      "fps": 30,
      "source_data": str(human),
      "robot_name": "unitree_g1",
      "source_sequence_key": "run",
      "frame_ids": np.arange(12),
    }
    self.path = self.root / "result.npz"

  def load(self) -> tuple[np.ndarray, dict]:
    np.savez(self.path, **self.payload)
    return read_result(self.path, self.source)

  def test_joint_reordering_and_quaternion_layout(self) -> None:
    values, provenance = self.load()
    np.testing.assert_array_equal(values[0, 7:], np.arange(29)[::-1])
    np.testing.assert_array_equal(values[0, 3:7], [0, 0, 0, 1])
    self.assertEqual(provenance["source_bvh_sha256"], self.source["sha256"])

  def test_rejects_invalid_result_contracts(self) -> None:
    for field, value in (
      ("robot_name", "cbot"),
      ("fps", 50),
      ("source_sequence_key", "other"),
      ("frame_ids", np.arange(1, 13)),
      ("robot_joint_names", np.array(["bad"] * 29)),
      ("qpos", np.full((12, 36), np.nan)),
      ("qpos", np.zeros((12, 36))),
    ):
      with self.subTest(field=field):
        original = self.payload[field]
        self.payload[field] = value
        with self.assertRaises(ValueError):
          self.load()
        self.payload[field] = original

  def test_rejects_changed_bvh(self) -> None:
    Path(self.source["bvh"]).write_text("changed")
    with self.assertRaisesRegex(ValueError, "BVH hash"):
      self.load()


if __name__ == "__main__":
  unittest.main()
