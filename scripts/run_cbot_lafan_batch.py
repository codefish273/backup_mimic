"""Supervise UMR retargeting, per-motion checks, CSV/NPZ conversion and statistics.

Launch from UMR's uv CUDA environment. Each phase writes durable progress so a
conversation interruption does not lose the batch state. UMR handles result
reuse; the checker validates output hashes before reusing checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def save(path, data):
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(data, indent=2) + "\n")
  temporary.replace(path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workspace", type=Path, default=Path("/home/zirui/mimic_clai"))
  parser.add_argument("--umr", type=Path, default=Path("/home/zirui/下载/UMR"))
  args = parser.parse_args()
  workspace, umr = args.workspace.resolve(), args.umr.resolve()
  root = workspace / "outputs/umr_lafan40"
  source_rows = json.loads((root / "source_manifest.json").read_text())
  if len(source_rows) != 40:
    raise ValueError("Expected exactly the selected 40 LAFAN motions")
  for row in source_rows:
    if not (root / "smplx" / f"{row['motion']}.npz").exists():
      raise FileNotFoundError(row["motion"])
  cpu_python = workspace / ".venv/bin/python"
  env = os.environ.copy()
  env.update(
    MPLCONFIGDIR="/tmp/mimic_lafan_mpl",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    PYTHONUNBUFFERED="1",
  )
  status = {
    "stage": "retargeting",
    "total_motions": 40,
    "total_frames": sum(r["frames"] for r in source_rows),
    "supervisor_pid": os.getpid(),
    "started_at": time.time(),
  }
  save(root / "run_status.json", status)
  batch_cmd = [
    sys.executable,
    str(umr / "scripts/humanoid_retarget_pipeline_batch.py"),
    "--config",
    str(workspace / "configs/umr/cbot_lafan40.json"),
    "--batch-config",
    str(workspace / "configs/umr/cbot_lafan40_batch.json"),
  ]
  check_cmd = [
    str(cpu_python),
    "-u",
    "-m",
    "scripts.audit_umr_batch",
    "--root",
    str(root),
    "--baseline-root",
    str(workspace / "outputs/lafan_audit_2026-09-10"),
    "--watch",
  ]
  with (
    (root / "retarget_batch.log").open("a") as batch_log,
    (root / "motion_checks.log").open("a") as check_log,
  ):
    batch = subprocess.Popen(
      batch_cmd, cwd=umr, env=env, stdout=batch_log, stderr=subprocess.STDOUT
    )
    checker = subprocess.Popen(
      check_cmd, cwd=workspace, env=env, stdout=check_log, stderr=subprocess.STDOUT
    )
    status.update(retarget_pid=batch.pid, checker_pid=checker.pid)
    raw_root = root / "retarget/cbot_v3_pelvis_flatskin"
    try:
      while batch.poll() is None:
        status["raw_results"] = len(list(raw_root.glob("*.npz")))
        checks = (
          json.loads((root / "motion_checks.json").read_text())
          if (root / "motion_checks.json").exists()
          else {}
        )
        status["checked_results"] = sum(
          v["status"] == "checked" for v in checks.values()
        )
        status["updated_at"] = time.time()
        if checker.poll() is not None:
          checker = subprocess.Popen(
            check_cmd,
            cwd=workspace,
            env=env,
            stdout=check_log,
            stderr=subprocess.STDOUT,
          )
          status["checker_pid"] = checker.pid
        save(root / "run_status.json", status)
        print(
          f"PROGRESS raw={status['raw_results']}/40 checked={status['checked_results']}/40",
          flush=True,
        )
        time.sleep(30)
      status["batch_returncode"] = batch.returncode
      checker.wait(timeout=300)
      if batch.returncode != 0 or checker.returncode != 0:
        raise RuntimeError(
          f"Batch/checker return codes: {batch.returncode}/{checker.returncode}"
        )
      status["stage"] = "csv_to_npz"
      status["raw_results"] = status["checked_results"] = 40
      save(root / "run_status.json", status)
      phases = [
        (
          "csv_to_npz",
          [
            str(cpu_python),
            "-u",
            str(workspace / "scripts/csv_to_npz.py"),
            "--robot",
            "cbot",
            "--input-dir",
            str(root / "csv"),
            "--output-dir",
            str(root / "smp_npz"),
            "--output-fps",
            "50",
            "--fk-backend",
            "mujoco",
          ],
        ),
        (
          "norm_stats",
          [
            str(cpu_python),
            "-u",
            str(workspace / "scripts/compute_norm_stats.py"),
            "--input-dir",
            str(root / "smp_npz"),
            "--output",
            str(root / "norm_stats.npz"),
          ],
        ),
      ]
      for stage, command in phases:
        status["stage"] = stage
        save(root / "run_status.json", status)
        with (root / f"{stage}.log").open("w") as log:
          subprocess.run(
            command,
            cwd=workspace,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
          )
      import numpy as np

      manifest = []
      for source in source_rows:
        path = root / "smp_npz" / f"{source['motion']}.npz"
        with np.load(path, allow_pickle=False) as data:
          windows = data["windows"]
          if not np.isfinite(windows).all() or windows.shape[1:] != (10, 59):
            raise ValueError(f"Invalid SMP windows: {path}")
          if (
            int(data["input_frame_count"]) != source["frames"]
            or float(data["input_fps"]) != 30
            or float(data["fps"][0]) != 50
          ):
            raise ValueError(f"Timing mismatch: {path}")
          count = int(data["resampled_frame_count"])
          if len(windows) != count - 9:
            raise ValueError(f"Window count mismatch: {path}")
          manifest.append(
            {
              "motion": source["motion"],
              "source_bvh_sha256": source["sha256"],
              "source_frames": source["frames"],
              "source_fps": 30,
              "fps": 50,
              "windows": len(windows),
              "file": str(path),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
          )
      save(root / "output_manifest.json", manifest)
      checks = json.loads((root / "motion_checks.json").read_text())
      status.update(
        stage="complete",
        total_windows=sum(x["windows"] for x in manifest),
        flagged_motions={
          name: item["flags"] for name, item in checks.items() if item["flags"]
        },
        finished_at=time.time(),
      )
      save(root / "run_status.json", status)
      print(
        "COMPLETE: 40 retargets, motion checks, CSVs, SMP NPZs and normalization statistics.",
        flush=True,
      )
    except BaseException as exc:
      status.update(stage="failed", error=repr(exc), updated_at=time.time())
      save(root / "run_status.json", status)
      raise


if __name__ == "__main__":
  main()
