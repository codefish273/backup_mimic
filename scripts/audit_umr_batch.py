"""Validate and inspect each completed UMR result, optionally while a batch runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scripts.audit_lafan import RobotAudit, quantiles
from scripts.csv_to_npz import JOINT_NAMES, _read_csv_motion
from scripts.umr_cbot_to_csv import Cfg as ExportCfg
from scripts.umr_cbot_to_csv import main as export_csv

from smp.rl.assets.cbotv3.cbot_v3 import CBOT_V3_NOHEAD_JOINT_NAMES


def write_json(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2) + "\n")
  temporary.replace(path)


def inspect(path, source, models, baseline_root, output_root):
  with np.load(path, allow_pickle=True) as result:
    qpos = result["qpos"]
    names = list(result["robot_joint_names"])
    frame_ids = result["frame_ids"]
    fps = float(np.asarray(result["fps"]).reshape(-1)[0])
    sequence = str(result["source_sequence_key"].item())
    source_smplx = Path(str(result["source_data"].item()))
  name, n = source["motion"], source["frames"]
  if qpos.shape != (n, 36) or not np.isfinite(qpos).all():
    raise ValueError(f"Expected {n} finite 36-DoF qpos frames; got {qpos.shape}")
  if sequence != name or fps != 30 or not np.array_equal(frame_ids, np.arange(n)):
    raise ValueError("Sequence identity, FPS or frame IDs changed")
  if len(names) != 29 or set(names) != set(CBOT_V3_NOHEAD_JOINT_NAMES):
    raise ValueError("Robot joint contract does not match CBot")
  if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-4):
    raise ValueError("Root quaternions are not normalized")
  if not source_smplx.is_file():
    raise ValueError(f"Missing source SMPL-X: {source_smplx}")
  with np.load(source_smplx, allow_pickle=False) as human:
    if str(human["source_sha256"].item()) != source["sha256"]:
      raise ValueError("Result does not refer to the selected BVH")
  joints = qpos[:, [7 + names.index(j) for j in CBOT_V3_NOHEAD_JOINT_NAMES]]
  values = np.column_stack((qpos[:, :3], qpos[:, 4:7], qpos[:, 3], joints))
  report = {
    "motion": name,
    "frames": n,
    "fps": fps,
    "duration_s": (n - 1) / fps,
    "file": str(path),
    "source_bvh": source["bvh"],
    "source_sha256": source["sha256"],
    "source_smplx": str(source_smplx),
    "finite_and_schema_valid": True,
    "joint_step_rad": quantiles(np.abs(np.diff(joints, axis=0))),
    "root_xy_speed_mps": quantiles(
      np.linalg.norm(np.gradient(qpos[:, :2], 1 / fps, axis=0), axis=-1)
    ),
    "flags": [],
  }
  frame_outputs = {}
  for label, model in models.items():
    metrics, arrays = model.run(values, fps)
    report[label] = metrics
    frame_outputs[label] = arrays
    out = output_root / "checks" / "frames" / f"{name}_{label}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
  g1_values, _ = _read_csv_motion(
    Path("datasets/csv/lafan") / f"{name}.csv", JOINT_NAMES, None
  )
  with np.load(baseline_root / "frames" / f"{name}_g1_training.npz") as baseline:
    support = (
      (baseline["foot_bottom_m"].min(axis=1) <= 0.01)
      & (baseline["root_up_z"] > 0.8)
      & (g1_values[:, 2] > 0.4)
    )
  foot_min = frame_outputs["cbot_training"]["foot_bottom_m"].min(axis=1)
  report["g1_upright_support_frames"] = int(support.sum())
  report["foot_height_when_g1_supported_m"] = quantiles(foot_min[support])
  report["above_3cm_when_g1_supported_fraction"] = (
    float(np.mean(foot_min[support] > 0.03)) if support.any() else None
  )
  train = report["cbot_training"]
  proxy = report["cbot_expanded_proxy"]
  if (
    report["above_3cm_when_g1_supported_fraction"] is not None
    and report["above_3cm_when_g1_supported_fraction"] > 0.25
  ):
    report["flags"].append("foot_clearance_on_source_support")
  if train["joint_limit_violation_frame_fraction"] > 0.001:
    report["flags"].append("joint_limit_violation")
  if np.mean(frame_outputs["cbot_training"]["ground_penetration_m"] > 0.01) > 0.01:
    report["flags"].append("ground_penetration_over_1cm")
  if (
    np.mean(frame_outputs["cbot_expanded_proxy"]["self_nontrunk_penetration_m"] > 0.02)
    > 0.01
  ):
    report["flags"].append("nontrunk_proxy_penetration_over_2cm")
  if proxy["self_contact_fraction"] > 0.1:
    report["flags"].append("inspect_proxy_contact_pairs")
  if report["joint_step_rad"]["max"] > 0.5:
    report["flags"].append("joint_step_over_half_radian")
  csv_path = output_root / "csv" / f"{name}.csv"
  export_csv(ExportCfg(path, csv_path))
  report["csv"] = str(csv_path)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path("outputs/umr_lafan40"))
  parser.add_argument(
    "--baseline-root", type=Path, default=Path("outputs/lafan_audit_2026-09-10")
  )
  parser.add_argument("--watch", action="store_true")
  parser.add_argument("--timeout-seconds", type=float, default=14400)
  args = parser.parse_args()
  torch.set_num_threads(1)
  sources = json.loads((args.root / "source_manifest.json").read_text())
  models = {
    "cbot_training": RobotAudit("cbot"),
    "cbot_expanded_proxy": RobotAudit("cbot", True),
  }
  summary_path = args.root / "motion_checks.json"
  summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
  fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  raw_root = args.root / "retarget" / "cbot_v3_pelvis_flatskin"
  start = time.monotonic()
  while True:
    for source in sources:
      name = source["motion"]
      path = raw_root / f"{name}_cbot_v3_pelvis_flatskin.npz"
      if not path.exists() or time.time() - path.stat().st_mtime < 2:
        continue
      digest = hashlib.sha256(path.read_bytes()).hexdigest()
      existing = summary.get(name, {})
      if (
        existing.get("npz_sha256") == digest
        and existing.get("checker_sha256") == fingerprint
      ):
        continue
      try:
        report = inspect(path, source, models, args.baseline_root, args.root)
        report["status"] = "checked"
      except Exception as exc:
        report = {
          "motion": name,
          "status": "invalid",
          "error": repr(exc),
          "frames": source["frames"],
          "flags": ["invalid_result"],
        }
      report.update(npz_sha256=digest, checker_sha256=fingerprint)
      summary[name] = report
      write_json(args.root / "checks" / f"{name}.json", report)
      write_json(summary_path, summary)
      print(
        f"CHECK {len(summary)}/{len(sources)} {name}: {report['status']}, flags={report['flags']}",
        flush=True,
      )
    rows = []
    for source in sources:
      item = summary.get(source["motion"])
      if item is None:
        continue
      rows.append(
        {
          "motion": source["motion"],
          "status": item["status"],
          "frames": item["frames"],
          "foot_height_median_m": item.get("cbot_training", {})
          .get("minimum_foot_height_m", {})
          .get("median"),
          "both_feet_above_2cm_fraction": item.get("cbot_training", {}).get(
            "both_feet_above_2cm_fraction"
          ),
          "self_contact_fraction_training": item.get("cbot_training", {}).get(
            "self_contact_fraction"
          ),
          "self_contact_fraction_proxy": item.get("cbot_expanded_proxy", {}).get(
            "self_contact_fraction"
          ),
          "flags": ";".join(item["flags"]),
        }
      )
    if rows:
      with (args.root / "motion_checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len(summary) == len(sources):
      break
    if not args.watch or time.monotonic() - start > args.timeout_seconds:
      break
    batch_summary = raw_root / "batch_summary.json"
    if batch_summary.exists():
      # Wait for a just-written final NPZ to pass the age guard above.
      if not all(
        (raw_root / f"{r['motion']}_cbot_v3_pelvis_flatskin.npz").exists()
        for r in sources
      ):
        break
    time.sleep(5)
  checked = sum(row["status"] == "checked" for row in summary.values())
  print(
    f"DONE checked={checked}/{len(sources)} frames={sum(r['frames'] for r in summary.values() if r['status'] == 'checked')}",
    flush=True,
  )
  if checked != len(sources):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
