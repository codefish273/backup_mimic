"""Compute per-feature q01/q99 quantiles from all NPZ window files.

Scans all ``*.npz`` files, stages windows on disk, and computes exact 1st/99th
percentiles one feature at a time with bounded memory. Saves the
result as a small ``.npz`` file for use by the training loop and RL reward.

Usage:
  uv run scripts/compute_norm_stats.py --input-dir datasets/npz --output datasets/norm_stats.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tyro


@dataclass
class Cfg:
  input_dir: str = "datasets/npz"
  """Directory containing windowed NPZ files."""
  output: str = "datasets/norm_stats.npz"
  """Output path for the quantile stats file."""
  q_low: float = 0.01
  """Lower quantile (default 1st percentile)."""
  q_high: float = 0.99
  """Upper quantile (default 99th percentile)."""


def main(cfg: Cfg) -> None:
  in_dir = Path(cfg.input_dir)
  npz_files = sorted(in_dir.glob("*.npz"))
  if not npz_files:
    raise FileNotFoundError(f"No NPZ files in {in_dir}")

  shapes: list[tuple[int, int]] = []
  for f in npz_files:
    with np.load(f, allow_pickle=False) as data:
      windows = data["windows"]  # (N, W, F)
      # Flatten window dim → treat each frame independently.
      shapes.append((windows.shape[0] * windows.shape[1], windows.shape[2]))
      print(f"  {f.name}: {windows.shape[0]} windows, {windows.shape[-1]} features")

  feature_dim = shapes[0][1]
  if any(shape[1] != feature_dim for shape in shapes):
    raise ValueError("NPZ files have inconsistent feature dimensions")
  num_frames = sum(shape[0] for shape in shapes)
  print(f"\nTotal frames: {num_frames}, feature dim: {feature_dim}")
  # Keep exact quantiles without retaining the entire dataset in RAM.
  with TemporaryDirectory(prefix="smp_norm_stats_") as temporary:
    all_frames = np.memmap(
      Path(temporary) / "frames.bin",
      mode="w+",
      dtype=np.float32,
      shape=(num_frames, feature_dim),
    )
    offset = 0
    for path, (count, _) in zip(npz_files, shapes, strict=True):
      with np.load(path, allow_pickle=False) as data:
        all_frames[offset : offset + count] = data["windows"].reshape(
          count, feature_dim
        )
      offset += count
    all_frames.flush()
    q_low = np.empty(feature_dim, dtype=np.float32)
    q_high = np.empty(feature_dim, dtype=np.float32)
    for index in range(feature_dim):
      q_low[index], q_high[index] = np.percentile(
        all_frames[:, index].astype(np.float64),
        [cfg.q_low * 100, cfg.q_high * 100],
      )
    del all_frames

  # Prevent zero-range (constant features) → set a minimum span.
  span = q_high - q_low
  tiny = span < 1e-6
  if tiny.any():
    print(
      f"  WARNING: {tiny.sum()} features have near-zero range, using fallback span=1.0"
    )
    q_high[tiny] = q_low[tiny] + 1.0

  out_path = Path(cfg.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out_path, q_low=q_low, q_high=q_high)
  print(f"\nSaved {out_path}: q_low/q_high shape ({q_low.shape[0]},)")
  print(f"  q_low  range: [{q_low.min():.4f}, {q_low.max():.4f}]")
  print(f"  q_high range: [{q_high.min():.4f}, {q_high.max():.4f}]")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
