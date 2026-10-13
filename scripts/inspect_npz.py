"""Inspect the structure of a NumPy NPZ archive.

Usage:
  uv run scripts/inspect_npz.py --input path/to/file.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Cfg:
  input: Path
  """Path to the NPZ file to inspect."""
  allow_pickle: bool = False
  """Whether to allow object arrays when loading the archive."""
  preview_items: int = 3
  """Number of items to preview for 1D object arrays or scalar-like values."""


def _describe_value(value: np.ndarray | object, preview_items: int) -> str:
  if isinstance(value, np.ndarray):
    if value.ndim == 0:
      return f"scalar {value.item()!r}"
    if value.dtype == object:
      flat = value.reshape(-1)
      preview = ", ".join(repr(item) for item in flat[:preview_items].tolist())
      suffix = "" if flat.size <= preview_items else ", ..."
      return f"object array shape={value.shape}, preview=[{preview}{suffix}]"
    if value.ndim == 1 and value.size <= preview_items:
      return f"values={value.tolist()}"
    return f"shape={value.shape}, dtype={value.dtype}"
  return repr(value)


def main(cfg: Cfg) -> None:
  if not cfg.input.exists():
    raise FileNotFoundError(cfg.input)

  try:
    data_cm = np.load(cfg.input, allow_pickle=cfg.allow_pickle)
    allow_pickle_used = cfg.allow_pickle
  except ValueError:
    if cfg.allow_pickle:
      raise
    data_cm = np.load(cfg.input, allow_pickle=True)
    allow_pickle_used = True

  with data_cm as data:
    keys = list(data.files)
    print(f"NPZ: {cfg.input}")
    print(f"Keys: {keys}")
    print(f"allow_pickle: {allow_pickle_used}")
    if allow_pickle_used and not cfg.allow_pickle:
      print("NOTE: object arrays were present, so the file was reloaded with allow_pickle=True")
    print()

    for key in keys:
      value = data[key]
      print(f"[{key}]")
      print(f"  type: {type(value).__name__}")
      if isinstance(value, np.ndarray):
        print(f"  shape: {value.shape}")
        print(f"  ndim: {value.ndim}")
        print(f"  dtype: {value.dtype}")
        print(f"  size: {value.size}")
        if value.ndim >= 1 and value.shape[0] > 0:
          print(f"  first item: {_describe_value(value[0], cfg.preview_items)}")
          if value.shape[0] > 1:
            print(f"  last item: {_describe_value(value[-1], cfg.preview_items)}")
      else:
        print(f"  value: {_describe_value(value, cfg.preview_items)}")
      print()


if __name__ == "__main__":
  main(tyro.cli(Cfg))