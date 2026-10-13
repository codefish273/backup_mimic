"""Convert official UMR G1 results and audit against the current training model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from scripts.csv_to_npz import EE_BODY_NAMES, JOINT_NAMES, _compute_windows

from smp.rl.env_cfg import g1_smp_env_cfg


def digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(".json.tmp")
  temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
  temporary.replace(path)


def read_result(path: Path, source: dict) -> tuple[np.ndarray, dict]:
  with np.load(path, allow_pickle=True) as result:
    qpos = result["qpos"]
    names = list(result["robot_joint_names"])
    fps = float(np.asarray(result["fps"]).reshape(-1)[0])
    source_path = Path(str(result["source_data"].item()))
    if str(result["robot_name"].item()) != "unitree_g1":
      raise ValueError("Expected unitree_g1 result")
    if str(result["source_sequence_key"].item()) != source["motion"]:
      raise ValueError("Source sequence mismatch")
    if not np.array_equal(result["frame_ids"], np.arange(source["frames"])):
      raise ValueError("Source frame IDs mismatch")
  if len(names) != 29 or set(names) != set(JOINT_NAMES):
    raise ValueError("G1 joint name contract mismatch")
  if qpos.shape != (source["frames"], 36) or not np.isfinite(qpos).all():
    raise ValueError("Invalid qpos shape or nonfinite values")
  if fps != 30 or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-4):
    raise ValueError("Invalid FPS or quaternion norm")
  if digest(Path(source["bvh"])) != source["sha256"]:
    raise ValueError("Source BVH hash mismatch")
  with np.load(source_path, allow_pickle=False) as human:
    if str(human["source_sha256"].item()) != source["sha256"]:
      raise ValueError("SMPL-X provenance mismatch")
    if len(human["poses"]) != source["frames"]:
      raise ValueError("SMPL-X frame count mismatch")
  joints = qpos[:, [7 + names.index(name) for name in JOINT_NAMES]]
  values = np.column_stack((qpos[:, :3], qpos[:, 4:7], qpos[:, 3], joints))
  return values, {
    "source_smplx": str(source_path),
    "source_smplx_sha256": digest(source_path),
    "source_bvh_sha256": source["sha256"],
    "result_sha256": digest(path),
  }


def audit(model: mujoco.MjModel, poses: np.ndarray) -> tuple[dict, dict, np.ndarray]:
  data = mujoco.MjData(model)
  joint_ids = [model.joint(f"robot/{name}").id for name in JOINT_NAMES]
  addresses = model.jnt_qposadr[joint_ids]
  body_ids = [model.body(f"robot/{name}").id for name in EE_BODY_NAMES]
  foot_ids = [
    np.array(
      [
        index
        for index in range(model.ngeom)
        if model.geom(index).name.startswith(f"robot/{side}_foot")
        and (model.geom_contype[index] or model.geom_conaffinity[index])
      ]
    )
    for side in ("left", "right")
  ]
  for indices in foot_ids:
    if len(indices) == 0 or not np.all(
      model.geom_type[indices] == mujoco.mjtGeom.mjGEOM_CAPSULE
    ):
      raise ValueError("Expected capsule foot collision geometry")
  frames = len(poses)
  arrays = {
    "foot_bottom_m": np.empty((frames, 2), dtype=np.float32),
    "ground_penetration_m": np.zeros(frames, dtype=np.float32),
    "self_penetration_m": np.zeros(frames, dtype=np.float32),
    "joint_limit_violation": np.zeros(frames, dtype=bool),
  }
  positions = np.empty((frames, len(body_ids), 3), dtype=np.float32)
  pairs: Counter = Counter()
  limits = model.jnt_range[joint_ids]
  for index, pose in enumerate(poses):
    data.qpos[:3] = pose[:3]
    data.qpos[3:7] = pose[[6, 3, 4, 5]]
    data.qpos[addresses] = pose[7:]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_collision(model, data)
    positions[index] = data.xpos[body_ids]
    arrays["joint_limit_violation"][index] = np.any(
      (pose[7:] < limits[:, 0] - 1e-3) | (pose[7:] > limits[:, 1] + 1e-3)
    )
    for side, indices in enumerate(foot_ids):
      arrays["foot_bottom_m"][index, side] = np.min(
        data.geom_xpos[indices, 2]
        - model.geom_size[indices, 0]
        - np.abs(data.geom_xmat[indices, 8]) * model.geom_size[indices, 1]
      )
    frame_pairs = set()
    for contact in data.contact:
      if contact.dist >= 0:
        continue
      body_names = [
        model.body(model.geom_bodyid[geom]).name
        for geom in (contact.geom1, contact.geom2)
      ]
      robot_bodies = [name.startswith("robot/") for name in body_names]
      if not any(robot_bodies):
        continue
      field = "self_penetration_m" if all(robot_bodies) else "ground_penetration_m"
      arrays[field][index] = max(arrays[field][index], -contact.dist)
      frame_pairs.add(" <-> ".join(sorted(body_names)))
    pairs.update(frame_pairs)
  summary = {
    "frames": frames,
    "minimum_foot_height_m": float(arrays["foot_bottom_m"].min()),
    "both_feet_above_2cm_fraction": float(
      np.mean(arrays["foot_bottom_m"].min(axis=1) > 0.02)
    ),
    "joint_limit_violation_fraction": float(arrays["joint_limit_violation"].mean()),
    "contact_pair_frame_counts": dict(pairs),
  }
  for field in ("ground_penetration_m", "self_penetration_m"):
    summary[field] = {
      "max": float(arrays[field].max()),
      "fraction_over_1cm": float(np.mean(arrays[field] > 0.01)),
      "worst_frame": int(arrays[field].argmax()),
    }
  return summary, arrays, positions


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path("outputs/umr_g1_lafan40"))
  parser.add_argument("--source-root", type=Path, default=Path("outputs/umr_lafan40"))
  parser.add_argument("--motion", action="append")
  args = parser.parse_args()
  torch.set_num_threads(2)
  sources = json.loads((args.source_root / "source_manifest.json").read_text())
  if args.motion:
    sources = [source for source in sources if source["motion"] in args.motion]
    if len(sources) != len(set(args.motion)):
      raise ValueError("Unknown source selection")
  model = Scene(g1_smp_env_cfg().scene, device="cpu").compile()
  for folder in ("csv", "smp_npz", "run_npz", "checks/frames"):
    (args.root / folder).mkdir(parents=True, exist_ok=True)
  records = []
  for source in sources:
    name = source["motion"]
    path = args.root / "retarget/unitree_g1" / f"{name}_unitree_g1.npz"
    values, provenance = read_result(path, source)
    csv_path = args.root / "csv" / f"{name}.csv"
    np.savetxt(
      csv_path,
      values,
      delimiter=",",
      fmt="%.10g",
      header="sample_rate: 30\ncolumns: root_xyz root_xyzw " + " ".join(JOINT_NAMES),
    )
    motion = MotionLoader(str(csv_path), 30, 50, "cpu")
    raw_summary, raw_arrays, _ = audit(model, values)
    resampled = np.column_stack(
      (
        motion.motion_base_poss.numpy(),
        motion.motion_base_rots.numpy()[:, [1, 2, 3, 0]],
        motion.motion_dof_poss.numpy(),
      )
    )
    summary, arrays, positions = audit(model, resampled)
    windows = (
      _compute_windows(
        motion.motion_base_poss,
        motion.motion_base_rots,
        motion.motion_base_lin_vels,
        motion.motion_base_ang_vels,
        torch.from_numpy(positions),
        motion.motion_dof_poss,
        10,
        1,
      )
      .numpy()
      .astype(np.float32)
    )
    if (
      windows.shape != (motion.output_frames - 9, 10, 59)
      or not np.isfinite(windows).all()
    ):
      raise ValueError(f"Invalid windows: {name}")
    with np.load(
      args.source_root / "smp_npz" / f"{name}.npz", allow_pickle=False
    ) as baseline:
      if baseline["windows"].shape != windows.shape:
        raise ValueError(f"CBot window alignment mismatch: {name}")
    output = args.root / "smp_npz" / f"{name}.npz"
    np.savez_compressed(
      output,
      windows=windows,
      fps=np.array([50], dtype=np.float32),
      window_size=np.array([10]),
      stride=np.array([1]),
      input_fps=30,
      input_frame_count=len(values),
      resampled_frame_count=motion.output_frames,
      joint_names=np.array(JOINT_NAMES),
      ee_body_names=np.array(EE_BODY_NAMES),
      feature_dims=np.array([3, 6, 29, 15, 3, 3]),
      robot="g1",
      source_bvh_sha256=source["sha256"],
    )
    if name.startswith("run"):
      import shutil

      shutil.copy2(output, args.root / "run_npz" / output.name)
    np.savez_compressed(args.root / "checks/frames" / f"{name}_30hz.npz", **raw_arrays)
    np.savez_compressed(args.root / "checks/frames" / f"{name}_50hz.npz", **arrays)
    record = {
      "motion": name,
      "windows": len(windows),
      "source_frames": len(values),
      "resampled_frames": motion.output_frames,
      "provenance": provenance,
      "source_30hz": raw_summary,
      "training_50hz": summary,
      "max_joint_step_rad": float(np.abs(np.diff(values[:, 7:], axis=0)).max()),
      "output_sha256": digest(output),
      "flags": [],
    }
    if record["max_joint_step_rad"] > 0.5:
      record["flags"].append("joint_step_over_half_radian")
    for label, report in (("source_30hz", raw_summary), ("training_50hz", summary)):
      for field in ("ground_penetration_m", "self_penetration_m"):
        if report[field]["fraction_over_1cm"] > 0.01:
          record["flags"].append(f"{label}:{field}_over_1cm")
      if report["joint_limit_violation_fraction"] > 0:
        record["flags"].append(f"{label}:joint_limit_violation")
    write_json(args.root / "checks" / f"{name}.json", record)
    records.append(record)
    print(f"CHECKED {name}: {windows.shape}, flags={record['flags']}", flush=True)
  report_name = "selected_manifest.json" if args.motion else "dataset_manifest.json"
  write_json(
    args.root / report_name,
    {
      "motions": len(records),
      "total_windows": sum(row["windows"] for row in records),
      "run_windows": sum(
        row["windows"] for row in records if row["motion"].startswith("run")
      ),
      "records": records,
      "scope": "Offline geometry/schema audit only; no policy training or dynamic validation.",
    },
  )


if __name__ == "__main__":
  main()
