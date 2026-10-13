"""Regression checks for duration-preserving conversion and full contact auditing."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scripts.audit_lafan import RobotAudit
from scripts.compute_norm_stats import Cfg as NormCfg
from scripts.compute_norm_stats import main as compute_stats
from scripts.csv_to_npz import Cfg, _read_csv_motion, main
from scripts.umr_cbot_to_csv import Cfg as UmrCfg
from scripts.umr_cbot_to_csv import main as export_umr

from smp.rl.assets.cbotv3.cbot_v3 import CBOT_V3_NOHEAD_JOINT_NAMES


class MotionDataRebuildTests(unittest.TestCase):
  def test_umr_export_reorders_joints_and_quaternion(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      names = list(reversed(CBOT_V3_NOHEAD_JOINT_NAMES))
      qpos = np.zeros((4, 36), dtype=np.float32)
      qpos[:, 2] = 1.0
      qpos[:, 3] = np.cos(0.2)
      qpos[:, 6] = np.sin(0.2)
      qpos[:, 7:] = np.arange(29) * 0.01
      source = root / "umr.npz"
      np.savez(
        source,
        qpos=qpos,
        robot_joint_names=np.array(names, dtype=object),
        fps=[30.0],
        robot_name="cbot_test",
      )
      destination = root / "motion.csv"
      export_umr(UmrCfg(source, destination))
      actual, fps = _read_csv_motion(destination, CBOT_V3_NOHEAD_JOINT_NAMES, None)
      self.assertEqual(fps, 30.0)
      np.testing.assert_array_equal(actual[:, 3:7], qpos[:, [4, 5, 6, 3]])
      np.testing.assert_array_equal(actual[:, 7:], qpos[:, 7:][:, ::-1])

  def test_cbot_override_preserves_duration_and_source_provenance(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      source_dir, output_dir = root / "csv", root / "npz"
      source_dir.mkdir()
      path = source_dir / "linear_motion.csv"
      with path.open("w", newline="") as stream:
        stream.write("# sample_rate: 50\n")
        writer = csv.writer(stream)
        writer.writerow(
          [
            "time",
            "root_x",
            "root_y",
            "root_z",
            "root_qx",
            "root_qy",
            "root_qz",
            "root_qw",
            *(f"dof_{name}" for name in CBOT_V3_NOHEAD_JOINT_NAMES),
          ]
        )
        for i in range(31):
          writer.writerow([i / 50, i / 30, 0, 1, 0, 0, 0, 1, *([0] * 29)])
      digest = hashlib.sha256(path.read_bytes()).hexdigest()
      main(
        Cfg(
          robot="cbot",
          input_dir=str(source_dir),
          output_dir=str(output_dir),
          input_fps=30,
          output_fps=50,
          fk_backend="mujoco",
        )
      )
      with np.load(output_dir / "linear_motion.npz", allow_pickle=False) as result:
        self.assertEqual(result["windows"].shape, (41, 10, 59))
        self.assertEqual(result["resampled_frame_count"].item(), 50)
        self.assertEqual(result["input_duration_s"].item(), 1.0)
        self.assertEqual(result["source_csv_sha256"].item(), digest)
        self.assertTrue(result["input_fps_overridden"].item())
        # A one-metre, one-second trajectory remains 1 m/s after resampling.
        np.testing.assert_allclose(result["windows"][..., 53], 1.0, atol=3e-6)
        np.testing.assert_allclose(result["windows"][..., 54:56], 0.0, atol=3e-6)
      self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

  def test_disk_backed_quantiles_match_in_memory_reference(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      data_dir = root / "data"
      data_dir.mkdir()
      rng = np.random.default_rng(42)
      chunks = [
        rng.normal(size=(7, 10, 5)).astype(np.float32),
        rng.normal(size=(9, 10, 5)).astype(np.float32),
      ]
      for index, chunk in enumerate(chunks):
        chunk[..., 2] = 3.0
        np.savez_compressed(data_dir / f"{index}.npz", windows=chunk)
      out = root / "stats.npz"
      compute_stats(NormCfg(input_dir=str(data_dir), output=str(out)))
      reference = np.concatenate(chunks).reshape(-1, 5).astype(np.float64)
      low, high = np.percentile(reference, [1, 99], axis=0).astype(np.float32)
      high[2] = low[2] + 1.0
      with np.load(out) as result:
        np.testing.assert_array_equal(result["q_low"], low)
        np.testing.assert_array_equal(result["q_high"], high)

  def test_terrain_body_is_not_classified_as_robot_self_contact(self):
    audit = RobotAudit("cbot")
    terrain_id = audit.model.geom("terrain").id
    self.assertGreater(audit.model.geom_bodyid[terrain_id], 0)
    self.assertFalse(audit.robot_geom[terrain_id])
    foot_id = audit.model.geom("robot/left_foot1_collision").id
    self.assertTrue(audit.robot_geom[foot_id])


if __name__ == "__main__":
  unittest.main()
