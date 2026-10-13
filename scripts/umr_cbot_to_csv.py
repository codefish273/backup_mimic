"""Export a trusted local UMR CBot result to the named CSV motion schema."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from smp.rl.assets.cbotv3.cbot_v3 import CBOT_V3_NOHEAD_JOINT_NAMES


@dataclass
class Cfg:
  input_file: Path
  output_file: Path


def main(cfg: Cfg) -> None:
  # Upstream UMR writes robot_joint_names as an object array.
  with np.load(cfg.input_file, allow_pickle=True) as result:
    qpos = result["qpos"]
    names = list(result["robot_joint_names"])
    fps = float(np.asarray(result["fps"]).reshape(-1)[0])
    robot = str(result["robot_name"].item())
  if not robot.startswith("cbot") or set(names) != set(CBOT_V3_NOHEAD_JOINT_NAMES):
    raise ValueError("Expected a CBot 29-joint UMR result")
  if len(names) != 29 or qpos.ndim != 2 or qpos.shape[1] != 36:
    raise ValueError("Expected qpos [root xyz, root wxyz, 29 named joints]")
  if len(qpos) < 3 or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
    raise ValueError("Motion must contain finite poses and positive FPS")
  if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-4):
    raise ValueError("Root quaternions must be normalized")
  ordered_joints = qpos[
    :, [7 + names.index(name) for name in CBOT_V3_NOHEAD_JOINT_NAMES]
  ]
  values = np.column_stack((qpos[:, :3], qpos[:, 4:7], qpos[:, 3], ordered_joints))
  cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
  with cfg.output_file.open("w", newline="") as stream:
    stream.write(f"# sample_rate: {fps:.9g}\n# source: {cfg.input_file}\n")
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
    for index, frame in enumerate(values):
      writer.writerow([index / fps, *frame.tolist()])
  print(f"Saved {cfg.output_file}: {len(values)} frames at {fps:g} FPS")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
