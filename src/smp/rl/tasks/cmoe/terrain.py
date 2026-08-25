# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).
#
# The terrain formulas below are adapted from CMoE/legged_gym and the
# BSD-licensed Isaac Gym terrain utilities.

"""The original CMoE heightfield terrain set for MJLab."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from scipy import interpolate

_HORIZONTAL_SCALE = 0.05
_VERTICAL_SCALE = 0.005
_DOWNSAMPLED_SCALE = 0.075
_TERRAIN_SIZE = (10.0, 10.0)
_ROUGH_HEIGHT = (0.01, 0.03)

# The original 40 terrain columns.  The first four columns are rough slopes,
# followed by the four stair-up, four stair-down, four discrete, twelve gap,
# four hurdle, four mixed-obstacle and four narrow-stair columns.
CMOE_COLUMN_KINDS = (
  ("rough_neg",) * 2
  + ("rough_pos",) * 2
  + ("stairs_up",) * 4
  + ("stairs_down",) * 4
  + ("discrete",) * 4
  + ("parkour_gap",) * 12
  + ("parkour_hurdle",) * 4
  + ("mix",) * 4
  + ("narrow_stairs",) * 4
)

# Values match HumanoidCfg.terrain.terrain_dict indexes.  This is kept next
# to the column layout so reward/termination code can use the original class.
CMOE_TERRAIN_TYPE_BY_COLUMN = np.array(
  [1] * 4
  + [2] * 4
  + [3] * 4
  + [4] * 4
  + [5] * 12
  + [8] * 4
  + [9] * 4
  + [10] * 4,
  dtype=np.int64,
)


def _random_uniform(raw: np.ndarray, difficulty: float, rng: np.random.Generator) -> None:
  max_height = (_ROUGH_HEIGHT[1] - _ROUGH_HEIGHT[0]) * difficulty + _ROUGH_HEIGHT[0]
  height = rng.uniform(_ROUGH_HEIGHT[0], max_height)
  min_height = int(-height / _VERTICAL_SCALE)
  max_height = int(height / _VERTICAL_SCALE)
  step = int(0.005 / _VERTICAL_SCALE)
  heights = np.arange(min_height, max_height + step, step)
  downsampled = rng.choice(
    heights,
    size=(
      int(raw.shape[0] * _HORIZONTAL_SCALE / _DOWNSAMPLED_SCALE),
      int(raw.shape[1] * _HORIZONTAL_SCALE / _DOWNSAMPLED_SCALE),
    ),
  )
  x = np.linspace(0, raw.shape[0] * _HORIZONTAL_SCALE, downsampled.shape[0])
  y = np.linspace(0, raw.shape[1] * _HORIZONTAL_SCALE, downsampled.shape[1])
  spline = interpolate.RectBivariateSpline(x, y, downsampled, kx=1, ky=1)
  xx = np.linspace(0, raw.shape[0] * _HORIZONTAL_SCALE, raw.shape[0])
  yy = np.linspace(0, raw.shape[1] * _HORIZONTAL_SCALE, raw.shape[1])
  raw += np.rint(spline(xx, yy)).astype(np.int16)


def _pyramid_slope(raw: np.ndarray, slope: float) -> None:
  width, length = raw.shape
  x = np.arange(width)
  y = np.arange(length)
  center_x = width // 2
  center_y = length // 2
  xx, yy = np.meshgrid(x, y, sparse=True)
  xx = ((center_x - np.abs(center_x - xx)) / center_x).reshape(width, 1)
  yy = ((center_y - np.abs(center_y - yy)) / center_y).reshape(1, length)
  max_height = int(slope * (_HORIZONTAL_SCALE / _VERTICAL_SCALE) * (width / 2))
  raw += (max_height * xx * yy).astype(raw.dtype)
  platform = int(3.0 / _HORIZONTAL_SCALE / 2)
  x1 = width // 2 - platform
  y1 = length // 2 - platform
  raw[:] = np.clip(raw, min(raw[x1, y1], 0), max(raw[x1, y1], 0))


def _pyramid_stairs(raw: np.ndarray, step_height: float) -> None:
  step_width = int(0.30 / _HORIZONTAL_SCALE)
  step_height = int(step_height / _VERTICAL_SCALE)
  platform = int(3.0 / _HORIZONTAL_SCALE)
  border = int(0.5 / _HORIZONTAL_SCALE)
  start_x, stop_x = border, raw.shape[0] - border
  start_y, stop_y = border, raw.shape[1] - border
  height = 0
  while (stop_x - start_x) > platform and (stop_y - start_y) > platform:
    start_x += step_width
    stop_x -= step_width
    start_y += step_width
    stop_y -= step_width
    height += step_height
    raw[start_x:stop_x, start_y:stop_y] = height


def _discrete_obstacles(raw: np.ndarray, difficulty: float, rng: np.random.Generator) -> None:
  max_height = int((0.05 + difficulty * 0.1) / _VERTICAL_SCALE)
  min_size = int(1.0 / _HORIZONTAL_SCALE)
  max_size = int(2.0 / _HORIZONTAL_SCALE)
  width_range = np.arange(min_size, max_size, 4)
  height_range = np.array([-max_height, -max_height // 2, max_height // 2, max_height])
  for _ in range(20):
    width = int(rng.choice(width_range))
    length = int(rng.choice(width_range))
    start_i = int(rng.choice(np.arange(0, raw.shape[0] - width, 4)))
    start_j = int(rng.choice(np.arange(0, raw.shape[1] - length, 4)))
    raw[start_i : start_i + width, start_j : start_j + length] = rng.choice(height_range)
  platform = int(3.0 / _HORIZONTAL_SCALE)
  x1, x2 = (raw.shape[0] - platform) // 2, (raw.shape[0] + platform) // 2
  y1, y2 = (raw.shape[1] - platform) // 2, (raw.shape[1] + platform) // 2
  raw[x1:x2, y1:y2] = 0


def _parkour_gap(raw: np.ndarray, difficulty: float, rng: np.random.Generator) -> None:
  mid_y = raw.shape[1] // 2
  platform_len = int(1.0 / _HORIZONTAL_SCALE)
  gap_depth = -int(rng.uniform(0.5, 1.5) / _VERTICAL_SCALE)
  half_valid_width = int(rng.uniform(1 - 0.5 * difficulty, 1.5 - 0.5 * difficulty) / _HORIZONTAL_SCALE)
  raw[:platform_len, :] = 0
  gap_size = int((0.1 + 0.7 * difficulty) / _HORIZONTAL_SCALE)
  dis_x_min = int(0.8 / _HORIZONTAL_SCALE) + gap_size
  dis_x_max = int(1.4 / _HORIZONTAL_SCALE) + gap_size
  dis_x = platform_len
  last_dis_x = dis_x
  for _ in range(4):
    dis_x += int(rng.integers(dis_x_min, dis_x_max))
    raw[dis_x - gap_size // 2 : dis_x + gap_size // 2, :] = gap_depth
    rand_y = int(rng.integers(-2, 2))
    raw[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = gap_depth
    raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width :] = gap_depth
    last_dis_x = dis_x
  pad = int(0.1 // _HORIZONTAL_SCALE)
  raw[:, :pad] = 0
  raw[:, -pad:] = 0
  raw[:pad, :] = 0
  raw[-pad:, :] = 0


def _parkour_hurdle(raw: np.ndarray, difficulty: float, rng: np.random.Generator) -> None:
  platform_len = int(1.0 / _HORIZONTAL_SCALE)
  stone_len = int((0.1 + 0.2 * difficulty) / _HORIZONTAL_SCALE)
  height_min = int((0.2 * difficulty) / _VERTICAL_SCALE)
  height_max = int((0.15 + 0.25 * difficulty) / _VERTICAL_SCALE)
  dis_x_min = int(1.2 / _HORIZONTAL_SCALE)
  dis_x_max = int(2.0 / _HORIZONTAL_SCALE)
  raw[:platform_len, :] = 0
  dis_x = platform_len
  for _ in range(4):
    dis_x += int(rng.integers(dis_x_min, dis_x_max))
    raw[dis_x - stone_len // 2 : dis_x + stone_len // 2, :] = rng.integers(
      height_min, height_max
    )
  pad = int(0.1 // _HORIZONTAL_SCALE)
  raw[:, :pad] = 0
  raw[:, -pad:] = 0
  raw[:pad, :] = 0
  raw[-pad:, :] = 0


def _mix_obstacles(raw: np.ndarray, difficulty: float, rng: np.random.Generator) -> None:
  diff = difficulty * 1.1
  gap_depth = -int(rng.integers(100, 300))
  raw[:40, :] = 0
  raw[30:36, :] = int(30 * diff)
  raw[36:42, :] = int(60 * diff)
  raw[42:48, :] = int(90 * diff)
  raw[48:60, :] = int(120 * diff)
  gap_start = 60
  gap_end = 72 - round(10 - diff * 10)
  raw[gap_start:gap_end, :] = gap_depth
  raw[gap_end:84, :] = int(120 * diff)
  raw[86:96, :] = int(96 * diff)
  raw[96:99, :] = int(170 * diff)
  raw[99:111, :] = int(120 * diff)
  gap_start = 111
  gap_end = 123 - round(10 - diff * 10)
  raw[gap_start:gap_end, :] = gap_depth
  raw[gap_end:140, :] = int(120 * diff)
  raw[140:160, :] = int(60 * diff)
  mid_y = raw.shape[1] // 2
  raw[:, mid_y + 20 :] = gap_depth
  raw[:, : mid_y - 20] = gap_depth


def _narrow_stairs(raw: np.ndarray, difficulty: float, rng: np.random.Generator) -> None:
  mid_y = raw.shape[1] // 2
  num_stones = 24
  step_height = int(0.25 * difficulty / _VERTICAL_SCALE)
  half_valid_width = int((1.0 - 0.5 * difficulty) / _HORIZONTAL_SCALE)
  platform_len = int(1.0 / _HORIZONTAL_SCALE)
  raw[:platform_len, :] = 0
  dis_x = platform_len
  stair_height = 0
  gap_depth = -int(rng.integers(10, 300))
  for i in range(num_stones):
    rand_x = 6
    if i < num_stones // 2 - 2:
      stair_height += step_height
    elif i > num_stones // 2 + 2:
      stair_height -= step_height
    raw[dis_x : dis_x + rand_x, :] = stair_height
    raw[dis_x : dis_x + rand_x, : mid_y - half_valid_width] = gap_depth
    raw[dis_x : dis_x + rand_x, mid_y + half_valid_width :] = gap_depth
    dis_x += rand_x
  pad = int(0.1 // _HORIZONTAL_SCALE)
  raw[:, :pad] = 0
  raw[:, -pad:] = 0
  raw[:pad, :] = 0
  raw[-pad:, :] = 0


def _make_raw(kind: str, difficulty: float, rng: np.random.Generator) -> np.ndarray:
  raw = np.zeros(
    (
      int(_TERRAIN_SIZE[0] / _HORIZONTAL_SCALE),
      int(_TERRAIN_SIZE[1] / _HORIZONTAL_SCALE),
    ),
    dtype=np.int16,
  )
  if kind.startswith("rough"):
    _pyramid_slope(raw, (-1 if kind == "rough_neg" else 1) * 0.4 * difficulty)
    _random_uniform(raw, difficulty, rng)
  elif kind == "stairs_up":
    _pyramid_stairs(raw, 0.05 + 0.18 * difficulty)
    _random_uniform(raw, difficulty, rng)
  elif kind == "stairs_down":
    _pyramid_stairs(raw, -(0.05 + 0.18 * difficulty))
    _random_uniform(raw, difficulty, rng)
  elif kind == "discrete":
    _discrete_obstacles(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  elif kind == "parkour_gap":
    _parkour_gap(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  elif kind == "parkour_hurdle":
    _parkour_hurdle(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  elif kind == "mix":
    _mix_obstacles(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  elif kind == "narrow_stairs":
    _narrow_stairs(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  return raw


def _heightfield_output(
  raw: np.ndarray,
  difficulty: float,
  spec: mujoco.MjSpec,
  spawn_at_center: bool,
) -> TerrainOutput:
  body = spec.body("terrain")
  minimum = int(raw.min())
  maximum = int(raw.max())
  elevation_range = max(maximum - minimum, 1)
  normalized = (raw - minimum) / elevation_range
  name = uuid.uuid4().hex
  field = spec.add_hfield(
    name=f"cmoe_hfield_{name}",
    size=[
      _TERRAIN_SIZE[0] / 2,
      _TERRAIN_SIZE[1] / 2,
      elevation_range * _VERTICAL_SCALE,
      max(elevation_range * _VERTICAL_SCALE, 0.05),
    ],
    nrow=raw.shape[0],
    ncol=raw.shape[1],
    userdata=normalized.astype(np.float32).flatten().tolist(),
  )
  material = spec.add_material(
    name=f"cmoe_terrain_material_{name}",
    rgba=(0.42, 0.46, 0.39, 1.0),
  )
  hfield_geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_HFIELD,
    hfieldname=field.name,
    pos=[_TERRAIN_SIZE[0] / 2, _TERRAIN_SIZE[1] / 2, minimum * _VERTICAL_SCALE],
    material=material.name,
  )
  if spawn_at_center:
    x1, x2 = int(4.0 / _HORIZONTAL_SCALE), int(6.0 / _HORIZONTAL_SCALE)
    y1, y2 = int(4.0 / _HORIZONTAL_SCALE), int(6.0 / _HORIZONTAL_SCALE)
    spawn_height = float(raw[x1:x2, y1:y2].max() * _VERTICAL_SCALE)
  else:
    spawn_height = 0.0
  spawn_x = _TERRAIN_SIZE[0] / 2 if spawn_at_center else 0.75
  return TerrainOutput(
    origin=np.array([spawn_x, _TERRAIN_SIZE[1] / 2, spawn_height]),
    geometries=[TerrainGeometry(geom=hfield_geom, hfield=field)],
  )


@dataclass(kw_only=True)
class CMoETerrainCfg(SubTerrainCfg):
  kind: str
  horizontal_scale: float = _HORIZONTAL_SCALE
  vertical_scale: float = _VERTICAL_SCALE

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    # TerrainGenerator adds a sub-row jitter. Recover the original row value.
    difficulty = np.floor(difficulty * 10.0) / 10.0
    raw = _make_raw(self.kind, difficulty, rng)
    return _heightfield_output(raw, difficulty, spec, self.kind not in {
      "parkour_gap",
      "parkour_hurdle",
      "mix",
      "narrow_stairs",
    })


def cmoe_terrain_generator_cfg() -> TerrainGeneratorCfg:
  sub_terrains = {
    f"{kind}_{column:02d}": CMoETerrainCfg(
      proportion=0.025,
      kind=kind,
    )
    for column, kind in enumerate(CMOE_COLUMN_KINDS)
  }
  return TerrainGeneratorCfg(
    seed=0,
    curriculum=True,
    size=_TERRAIN_SIZE,
    border_width=25.0,
    num_rows=10,
    num_cols=40,
    color_scheme="none",
    sub_terrains=sub_terrains,
    add_lights=True,
  )


__all__ = [
  "CMOE_COLUMN_KINDS",
  "CMOE_TERRAIN_TYPE_BY_COLUMN",
  "CMoETerrainCfg",
  "cmoe_terrain_generator_cfg",
]
