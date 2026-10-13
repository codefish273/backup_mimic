"""Audit paired robot CSVs at every source frame using actual MuJoCo geometry.

Outputs alignment evidence, per-frame foot/contact arrays, per-clip summaries,
contact body pairs, and collision coverage. The CBot expanded-proxy mode is a
diagnostic model, not a change to the training collision configuration.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
from mjlab.scene import Scene
from scripts.csv_to_npz import JOINT_NAMES, _read_csv_motion

from smp.rl.assets.cbotv3.cbot_v3 import CBOT_V3_NOHEAD_JOINT_NAMES
from smp.rl.env_cfg import cbot_smp_env_cfg, g1_smp_env_cfg


@dataclass
class Cfg:
  g1_dir: Path = Path("datasets/csv/lafan")
  cbot_dir: Path = Path("datasets/csv_cbot/lafan")
  bvh_dir: Path = Path("/home/zirui/gmr-motionlab/data/lafan")
  output_dir: Path = Path("outputs/lafan_audit_2026-09-10")
  alignment_only: bool = False
  expanded_cbot: bool = True


def dump_json(path: Path, value: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def quantiles(values: np.ndarray) -> dict[str, float | None]:
  if values.size == 0:
    return {
      key: None for key in ("min", "p01", "p05", "median", "p95", "p99", "max", "mean")
    }
  out = dict(
    zip(
      ("min", "p01", "p05", "median", "p95", "p99", "max"),
      np.quantile(values, (0, 0.01, 0.05, 0.5, 0.95, 0.99, 1)).tolist(),
      strict=True,
    )
  )
  out["mean"] = float(np.mean(values))
  return out


def alignment_record(gp: Path, cp: Path, bvh_dir: Path) -> dict:
  g, gf = _read_csv_motion(gp, JOINT_NAMES, None)
  c, cf = _read_csv_motion(cp, CBOT_V3_NOHEAD_JOINT_NAMES, None)
  record = {
    "motion": gp.stem,
    "g1_frames": len(g),
    "cbot_frames": len(c),
    "g1_input_fps": gf,
    "cbot_declared_fps": cf,
    "g1_csv": str(gp),
    "cbot_csv": str(cp),
    "g1_sha256": hashlib.sha256(gp.read_bytes()).hexdigest(),
    "cbot_sha256": hashlib.sha256(cp.read_bytes()).hexdigest(),
    "aligned": False,
  }
  bvh = bvh_dir / f"{gp.stem}.bvh"
  if bvh.is_file():
    with bvh.open() as source:
      for line in source:
        if line.startswith("Frames:"):
          record["bvh_frames"] = int(line.split(":")[1])
        if line.startswith("Frame Time:"):
          record["bvh_frame_time"] = float(line.split(":")[1])
          break
  if len(g) != len(c):
    return record
  dg, dc = np.diff(g[:, :2], axis=0), np.diff(c[:, :2], axis=0)
  scale = float((dg * dc).sum() / max(np.square(dg).sum(), 1e-12))
  residual = float(np.linalg.norm(dc - scale * dg) / max(np.linalg.norm(dc), 1e-12))
  record.update(
    xy_delta_scale=scale,
    relative_xy_delta_residual=residual,
    restored_input_fps=gf,
    restored_duration_s=(len(c) - 1) / gf,
    declared_duration_s=(len(c) - 1) / cf,
  )
  record["aligned"] = (
    1.15 < scale < 1.30
    and residual < 0.05
    and record.get("bvh_frames", len(g)) == len(g)
    and abs(record.get("bvh_frame_time", 1 / gf) * gf - 1) < 0.001
  )
  return record


def body_region(name: str) -> str:
  name = name.removeprefix("robot/")
  side = (
    "left" if name.startswith("left_") else "right" if name.startswith("right_") else ""
  )
  if "foot" in name or "ankle" in name:
    return f"{side}_foot"
  if any(token in name for token in ("wrist", "hand", "palm")):
    return f"{side}_hand"
  if any(token in name for token in ("shoulder", "elbow")):
    return f"{side}_arm"
  if any(token in name for token in ("hip", "knee", "thigh", "shin", "linkage")):
    return f"{side}_leg"
  if "head" in name:
    return "head"
  return "trunk"


def foot_bottoms(
  model: mujoco.MjModel, data: mujoco.MjData, ids: np.ndarray
) -> np.ndarray:
  """Exact world-z support of the capsule foot geoms used by both robots."""
  if not np.all(model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_CAPSULE):
    raise ValueError("Expected capsule foot collision geometry")
  return (
    data.geom_xpos[ids, 2]
    - model.geom_size[ids, 0]
    - np.abs(data.geom_xmat[ids, 8]) * model.geom_size[ids, 1]
  )


class RobotAudit:
  def __init__(self, robot: str, expanded: bool = False):
    cfg = cbot_smp_env_cfg() if robot == "cbot" else g1_smp_env_cfg()
    self.label = robot + ("_expanded_proxy" if expanded else "_training")
    if expanded:
      entity = cfg.scene.entities["robot"]
      entity.collisions = tuple(
        replace(
          collision,
          geom_names_expr=(".*_collision", ".*_zone.*", ".*_safety_zone.*"),
          condim={r"(left|right)_foot[1-8]_collision": 3, ".*": 1},
        )
        for collision in entity.collisions
      )
    self.model = Scene(cfg.scene, device="cpu").compile()
    self.data = mujoco.MjData(self.model)
    self.joint_names = CBOT_V3_NOHEAD_JOINT_NAMES if robot == "cbot" else JOINT_NAMES
    joints = [self.model.joint(f"robot/{name}").id for name in self.joint_names]
    self.joint_addresses = self.model.jnt_qposadr[joints]
    self.joint_limits = self.model.jnt_range[joints]
    self.robot_geom = np.array(
      [
        self.model.body(int(self.model.geom_bodyid[i])).name.startswith("robot/")
        for i in range(self.model.ngeom)
      ]
    )
    self.regions = [
      body_region(self.model.body(int(b)).name) for b in self.model.geom_bodyid
    ]
    self.feet = [
      np.array(
        [
          i
          for i in range(self.model.ngeom)
          if self.model.geom(i).name.startswith(f"robot/{side}_foot")
          and self.model.geom(i).name.endswith("_collision")
        ],
        dtype=int,
      )
      for side in ("left", "right")
    ]
    self.foot_bodies = [int(self.model.geom_bodyid[ids[0]]) for ids in self.feet]
    self.coverage = [
      {
        "geom_id": i,
        "geom": self.model.geom(i).name,
        "body": self.model.body(int(self.model.geom_bodyid[i])).name,
        "active": bool(self.model.geom_contype[i] or self.model.geom_conaffinity[i]),
        "type": int(self.model.geom_type[i]),
        "condim": int(self.model.geom_condim[i]),
        "region": self.regions[i],
      }
      for i in range(self.model.ngeom)
      if self.robot_geom[i]
    ]

  def run(self, values: np.ndarray, fps: float) -> tuple[dict, dict[str, np.ndarray]]:
    n = len(values)
    arrays = {
      "foot_bottom_m": np.empty((n, 2), dtype=np.float32),
      "foot_position_w": np.empty((n, 2, 3), dtype=np.float32),
      "foot_up_z": np.empty((n, 2), dtype=np.float32),
      "root_up_z": np.empty(n, dtype=np.float32),
      "ground_contact": np.zeros(n, dtype=bool),
      "nonfoot_ground_contact": np.zeros(n, dtype=bool),
      "self_contact": np.zeros(n, dtype=bool),
      "self_nontrunk_contact": np.zeros(n, dtype=bool),
      "ground_penetration_m": np.zeros(n, dtype=np.float32),
      "self_penetration_m": np.zeros(n, dtype=np.float32),
      "self_nontrunk_penetration_m": np.zeros(n, dtype=np.float32),
    }
    region_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    pair_depths: dict[str, float] = {}
    violations = (values[:, 7:] < self.joint_limits[:, 0] - 1e-3) | (
      values[:, 7:] > self.joint_limits[:, 1] + 1e-3
    )
    for index, frame in enumerate(values):
      self.data.qpos[:3] = frame[:3]
      self.data.qpos[3:7] = frame[[6, 3, 4, 5]]
      self.data.qpos[self.joint_addresses] = frame[7:]
      mujoco.mj_kinematics(self.model, self.data)
      mujoco.mj_comPos(self.model, self.data)
      mujoco.mj_collision(self.model, self.data)
      for side, ids in enumerate(self.feet):
        arrays["foot_bottom_m"][index, side] = foot_bottoms(
          self.model, self.data, ids
        ).min()
      arrays["foot_position_w"][index] = self.data.xpos[self.foot_bodies]
      arrays["foot_up_z"][index] = self.data.xmat[self.foot_bodies, 8]
      arrays["root_up_z"][index] = self.data.xmat[self.model.body("robot/pelvis").id, 8]
      regions, pairs = set(), set()
      for contact in self.data.contact:
        if contact.dist > 0:
          continue
        a, b = int(contact.geom1), int(contact.geom2)
        ra, rb = self.robot_geom[a], self.robot_geom[b]
        if not (ra or rb):
          continue
        depth = max(-float(contact.dist), 0.0)
        if ra and rb:
          arrays["self_contact"][index] = True
          if not (self.regions[a] == self.regions[b] == "trunk"):
            arrays["self_nontrunk_contact"][index] = True
            arrays["self_nontrunk_penetration_m"][index] = max(
              arrays["self_nontrunk_penetration_m"][index], depth
            )
          arrays["self_penetration_m"][index] = max(
            arrays["self_penetration_m"][index], depth
          )
          names = sorted(
            self.model.body(int(self.model.geom_bodyid[g])).name for g in (a, b)
          )
          pair = "self:" + " <-> ".join(names)
        else:
          robot_id = a if ra else b
          arrays["ground_contact"][index] = True
          region = self.regions[robot_id]
          regions.add(region)
          if region not in ("left_foot", "right_foot"):
            arrays["nonfoot_ground_contact"][index] = True
          arrays["ground_penetration_m"][index] = max(
            arrays["ground_penetration_m"][index], depth
          )
          pair = "ground:" + self.model.body(int(self.model.geom_bodyid[robot_id])).name
        pairs.add(pair)
        pair_depths[pair] = max(pair_depths.get(pair, 0.0), depth)
      region_counts.update(regions)
      pair_counts.update(pairs)
    minimum = arrays["foot_bottom_m"].min(axis=1)
    foot_speed = np.linalg.norm(
      np.gradient(arrays["foot_position_w"][..., :2], 1 / fps, axis=0), axis=-1
    )
    arrays["foot_xy_speed_mps"] = foot_speed
    near_ground = arrays["foot_bottom_m"] <= 0.01
    summary = {
      "frames": n,
      "minimum_foot_height_m": quantiles(minimum),
      "foot_height_m": [quantiles(arrays["foot_bottom_m"][:, i]) for i in (0, 1)],
      "both_feet_above_2cm_fraction": float(np.mean(minimum > 0.02)),
      "both_feet_above_4cm_fraction": float(np.mean(minimum > 0.04)),
      "foot_below_minus_1cm_fraction": float(np.mean(minimum < -0.01)),
      "ground_contact_fraction": float(arrays["ground_contact"].mean()),
      "nonfoot_ground_contact_fraction": float(arrays["nonfoot_ground_contact"].mean()),
      "self_contact_fraction": float(arrays["self_contact"].mean()),
      "self_nontrunk_contact_fraction": float(arrays["self_nontrunk_contact"].mean()),
      "ground_penetration_m": quantiles(arrays["ground_penetration_m"]),
      "self_penetration_m": quantiles(arrays["self_penetration_m"]),
      "self_nontrunk_penetration_m": quantiles(arrays["self_nontrunk_penetration_m"]),
      "near_ground_foot_xy_speed_mps": quantiles(foot_speed[near_ground]),
      "ground_contact_region_frame_counts": dict(region_counts),
      "contact_pairs": {
        p: {"frames": count, "max_depth_m": pair_depths[p]}
        for p, count in pair_counts.most_common()
      },
      "joint_limit_violation_frame_fraction": float(violations.any(axis=1).mean()),
      "joint_limit_violations": {
        name: int(count)
        for name, count in zip(self.joint_names, violations.sum(axis=0), strict=True)
        if count
      },
    }
    return summary, arrays


def main(cfg: Cfg) -> None:
  torch.set_num_threads(1)
  cfg.output_dir.mkdir(parents=True, exist_ok=True)
  gps = sorted(cfg.g1_dir.glob("*.csv"))
  cps = sorted(cfg.cbot_dir.glob("*.csv"))
  expected = {p.name.replace("g1_", "cbot_") for p in gps}
  if not gps or expected != {p.name for p in cps}:
    raise ValueError("Both directories must contain matching complete clip sets")
  alignment = [
    alignment_record(gp, cfg.cbot_dir / gp.name.replace("g1_", "cbot_"), cfg.bvh_dir)
    for gp in gps
  ]
  dump_json(cfg.output_dir / "alignment.json", alignment)
  with (cfg.output_dir / "alignment.csv").open("w", newline="") as stream:
    columns = sorted(set().union(*(r.keys() for r in alignment)))
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(alignment)
  if not all(row["aligned"] for row in alignment):
    raise ValueError("Some motions failed frame-alignment checks; see alignment.json")
  print(
    f"ALIGNMENT: {len(alignment)} clips, {sum(r['g1_frames'] for r in alignment)} frames, all passed",
    flush=True,
  )
  if cfg.alignment_only:
    return
  models = [RobotAudit("g1"), RobotAudit("cbot")]
  if cfg.expanded_cbot:
    models.append(RobotAudit("cbot", expanded=True))
  dump_json(
    cfg.output_dir / "collision_coverage.json", {m.label: m.coverage for m in models}
  )
  records, csv_rows = [], []
  for number, (gp, alignment_row) in enumerate(zip(gps, alignment, strict=True), 1):
    cp = cfg.cbot_dir / gp.name.replace("g1_", "cbot_")
    values = {
      "g1": _read_csv_motion(gp, JOINT_NAMES, None)[0],
      "cbot": _read_csv_motion(cp, CBOT_V3_NOHEAD_JOINT_NAMES, None)[0],
    }
    record = {"motion": gp.stem, "frames": len(values["g1"])}
    results = {}
    for model in models:
      summary, arrays = model.run(
        values["cbot" if model.label.startswith("cbot") else "g1"],
        alignment_row["g1_input_fps"],
      )
      record[model.label] = summary
      results[model.label] = arrays
      frame_path = cfg.output_dir / "frames" / f"{gp.stem}_{model.label}.npz"
      frame_path.parent.mkdir(exist_ok=True)
      np.savez_compressed(frame_path, **arrays)
      csv_rows.append(
        {
          "motion": gp.stem,
          "model": model.label,
          "frames": len(values["g1"]),
          "foot_height_median_m": summary["minimum_foot_height_m"]["median"],
          **{k: v for k, v in summary.items() if k.endswith("fraction")},
          "max_ground_penetration_m": summary["ground_penetration_m"]["max"],
          "max_self_penetration_m": summary["self_penetration_m"]["max"],
        }
      )
    ga, ca = results["g1_training"], results["cbot_training"]
    support = (
      (ga["foot_bottom_m"].min(axis=1) <= 0.01)
      & (ga["root_up_z"] > 0.8)
      & (values["g1"][:, 2] > 0.4)
    )
    record["g1_upright_support_frames"] = int(support.sum())
    record["cbot_minimum_foot_height_when_g1_supported_m"] = quantiles(
      ca["foot_bottom_m"].min(axis=1)[support]
    )
    record["paired_minimum_foot_height_delta_m"] = quantiles(
      ca["foot_bottom_m"].min(axis=1) - ga["foot_bottom_m"].min(axis=1)
    )
    records.append(record)
    dump_json(cfg.output_dir / "contacts.json", records)
    with (cfg.output_dir / "contacts.csv").open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
      writer.writeheader()
      writer.writerows(csv_rows)
    print(
      f"AUDIT {number}/{len(gps)} {gp.stem}: G1 foot={record['g1_training']['minimum_foot_height_m']['median']:.4f} CBot foot={record['cbot_training']['minimum_foot_height_m']['median']:.4f}",
      flush=True,
    )
  summarize(cfg.output_dir, records)


def summarize(output_dir: Path, records: list[dict]) -> dict:
  """Aggregate frame-weighted rates and exact foot-height quantiles."""
  labels = [key for key in records[0] if key.endswith(("_training", "_proxy"))]
  summary = {
    "clips": len(records),
    "paired_frames": sum(r["frames"] for r in records),
    "models": {},
  }
  families: dict[str, list[dict]] = {}
  for record in records:
    family = record["motion"].split("_subject", 1)[0].rstrip("0123456789")
    families.setdefault(family, []).append(record)
  family_rows = []
  for label in labels:
    frame_arrays = {}
    for record in records:
      with np.load(output_dir / "frames" / f"{record['motion']}_{label}.npz") as data:
        frame_arrays[record["motion"]] = {
          "minimum_foot_height_m": data["foot_bottom_m"].min(axis=1),
          **{
            key: data[key]
            for key in (
              "ground_contact",
              "nonfoot_ground_contact",
              "self_contact",
              "ground_penetration_m",
              "self_penetration_m",
            )
          },
        }

    def aggregate(subset, arrays_by_clip):
      arrays = {
        key: np.concatenate([arrays_by_clip[r["motion"]][key] for r in subset])
        for key in next(iter(arrays_by_clip.values()))
      }
      return {
        "frames": len(arrays["ground_contact"]),
        "minimum_foot_height_m": quantiles(arrays["minimum_foot_height_m"]),
        **{
          key + "_fraction": float(arrays[key].mean())
          for key in ("ground_contact", "nonfoot_ground_contact", "self_contact")
        },
        "both_feet_above_2cm_fraction": float(
          np.mean(arrays["minimum_foot_height_m"] > 0.02)
        ),
        "ground_penetration_over_1cm_fraction": float(
          np.mean(arrays["ground_penetration_m"] > 0.01)
        ),
        "self_penetration_over_2cm_fraction": float(
          np.mean(arrays["self_penetration_m"] > 0.02)
        ),
      }

    summary["models"][label] = aggregate(records, frame_arrays)
    for family, subset in families.items():
      row = aggregate(subset, frame_arrays)
      row["foot_height_median_m"] = row.pop("minimum_foot_height_m")["median"]
      family_rows.append({"family": family, "model": label, **row})
  summary["by_family"] = family_rows
  dump_json(output_dir / "summary.json", summary)
  with (output_dir / "by_family.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(family_rows[0]))
    writer.writeheader()
    writer.writerows(family_rows)
  return summary


if __name__ == "__main__":
  main(tyro.cli(Cfg))
