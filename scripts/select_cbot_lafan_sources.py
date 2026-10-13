"""Select LAFAN BVHs strictly from the current CBot CSV basenames."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from scripts.csv_to_npz import _read_csv_motion

from smp.rl.assets.cbotv3.cbot_v3 import CBOT_V3_NOHEAD_JOINT_NAMES


@dataclass
class Cfg:
  cbot_csv_dir: Path = Path("datasets/csv_cbot/lafan")
  bvh_dir: Path = Path("/home/zirui/下载/gmr-dev/lafan1")
  output_root: Path = Path("outputs/umr_lafan40")
  expected_count: int = 40


def main(cfg: Cfg) -> None:
  csv_files = sorted(cfg.cbot_csv_dir.glob("*.csv"))
  if len(csv_files) != cfg.expected_count:
    raise ValueError(f"Expected {cfg.expected_count} CBot CSVs, found {len(csv_files)}")
  rows = []
  for path in csv_files:
    source = cfg.bvh_dir / f"{path.stem}.bvh"
    if not source.is_file():
      raise FileNotFoundError(f"No same-name LAFAN BVH for {path}: {source}")
    values, declared_fps = _read_csv_motion(path, CBOT_V3_NOHEAD_JOINT_NAMES, None)
    frames, frame_time = None, None
    with source.open() as stream:
      for line in stream:
        if line.startswith("Frames:"):
          frames = int(line.split(":")[1])
        if line.startswith("Frame Time:"):
          frame_time = float(line.split(":")[1])
          break
    if frames != len(values) or frame_time is None or abs(frame_time * 30 - 1) > 0.001:
      raise ValueError(f"CBot CSV/BVH frame or timing mismatch: {path.stem}")
    rows.append(
      {
        "motion": path.stem,
        "selection_basis": "cbot_csv_basename",
        "cbot_csv": str(path.resolve()),
        "cbot_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cbot_csv_declared_fps": declared_fps,
        "bvh": str(source.absolute()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frames": frames,
        "frame_time": frame_time,
        "nominal_fps": 30,
      }
    )
  cfg.output_root.mkdir(parents=True, exist_ok=True)
  manifest = cfg.output_root / "source_manifest.json"
  if manifest.exists():
    old = json.loads(manifest.read_text())
    previous = {row["motion"]: row["sha256"] for row in old}
    current = {row["motion"]: row["sha256"] for row in rows}
    if previous != current:
      raise ValueError(
        "Existing batch selected different sources; refusing to relabel its results"
      )
  manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
  mapping_path = cfg.output_root / "cbot_csv_to_bvh_mapping.csv"
  with mapping_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(
    f"Matched {len(rows)} CBot CSV basenames to BVHs; {sum(r['frames'] for r in rows)} frames. Existing source hashes unchanged."
  )
  print(mapping_path)


if __name__ == "__main__":
  main(tyro.cli(Cfg))
