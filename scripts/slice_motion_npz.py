"""Slice the leading axis of an SMP motion npz.

  uv run scripts/slice_motion_npz.py \\
    lafan_run_cbot_pretrain/motion/run1_subject2.npz 900 3900
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def slice_npz(src: Path, start: int, stop: int, dst: Path | None = None) -> Path:
  if start < 0 or stop <= start:
    raise ValueError(f"invalid slice [{start}:{stop})")
  if dst is None:
    dst = src.with_name(f"{src.stem}_{start}_{stop}.npz")

  with np.load(src, allow_pickle=False) as data:
    windows = data["windows"]
    n = windows.shape[0]
    if stop > n:
      raise ValueError(f"stop {stop} exceeds windows length {n}")

    out: dict[str, np.ndarray] = {}
    for key in data.files:
      array = data[key]
      if key == "windows":
        out[key] = array[start:stop]
      elif key in ("resampled_frame_count", "input_frame_count", "input_duration_s"):
        continue
      else:
        out[key] = array

    sliced = out["windows"]
    fps = float(np.asarray(out["fps"]).reshape(-1)[0])
    window_size = int(np.asarray(out["window_size"]).reshape(-1)[0])
    stride = int(np.asarray(out["stride"]).reshape(-1)[0])
    resampled = (sliced.shape[0] - 1) * stride + window_size
    out["resampled_frame_count"] = np.array(resampled, dtype=np.int64)
    if "input_fps" in out:
      input_fps = float(np.asarray(out["input_fps"]))
      out["input_frame_count"] = np.array(
        int(round(resampled * input_fps / fps)), dtype=np.int64
      )
      out["input_duration_s"] = np.array(resampled / fps, dtype=np.float64)

  np.savez_compressed(dst, **out)
  return dst


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("src", type=Path)
  parser.add_argument("start", type=int)
  parser.add_argument("stop", type=int)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  dst = slice_npz(args.src, args.start, args.stop, args.output)
  with np.load(dst, allow_pickle=False) as data:
    print(
      f"saved {dst}: windows={data['windows'].shape} "
      f"resampled_frame_count={int(data['resampled_frame_count'])}"
    )


if __name__ == "__main__":
  main()
