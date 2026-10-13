"""Batch adapter for the local jaraujo98/lafan_to_smplx conversion.

Uses the upstream rotation offsets, joint map and fixed shape coefficients,
and GMR's BVH parser/FK utilities. SMPL-X evaluation is batched on CUDA. No robot
motion is used as an intermediate representation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES


def load_conversion_sources(upstream: Path, vendor_dir: Path):
  # The upstream module imports the entire GMR stack before exposing these
  # constants. Load its three declarations without robot/viewer dependencies.
  tree = ast.parse(upstream.read_text())
  wanted = {"lafan_frame_offsets", "smplx_to_lafan_map", "betas"}
  declarations = [
    node
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(
      isinstance(target, ast.Name) and target.id in wanted for target in node.targets
    )
  ]
  scope = {"np": np, "torch": torch, "R": R}
  exec(
    compile(ast.Module(body=declarations, type_ignores=[]), str(upstream), "exec"),
    scope,
  )
  if not wanted.issubset(scope):
    raise ValueError("Upstream conversion declarations changed")
  package = "_smp_lafan_vendor"
  spec = importlib.util.spec_from_file_location(
    package, vendor_dir / "__init__.py", submodule_search_locations=[str(vendor_dir)]
  )
  module = importlib.util.module_from_spec(spec)
  sys.modules[package] = module
  spec.loader.exec_module(module)
  extract = importlib.import_module(package + ".extract")
  utils = importlib.import_module(package + ".utils")
  return scope, extract, utils


def convert(path, body_model, scope, extract, utils, batch_size, device):
  data = extract.read_bvh(str(path))
  global_quat, global_pos = utils.quat_fk(data.quats, data.pos, data.parents)
  # The same LAFAN y-up -> z-up conversion used by GMR's human BVH reader.
  matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
  rotation = R.from_matrix(matrix).as_quat(scalar_first=True)
  global_quat = utils.quat_mul(rotation, global_quat)
  global_pos = (global_pos @ matrix.T / 100).astype(np.float32)
  n = len(global_pos)
  offsets, mapping = scope["lafan_frame_offsets"], scope["smplx_to_lafan_map"]
  rotations = {
    name: R.from_quat(global_quat[:, index], scalar_first=True) * offsets[name]
    for index, name in enumerate(data.bones)
  }
  poses = np.zeros((n, 55, 3), dtype=np.float32)
  poses[:, 0] = rotations["Hips"].as_rotvec()
  for index, name in enumerate(JOINT_NAMES[1:22], 1):
    target = mapping[name]
    target_id = data.bones.index(target)
    parent = data.bones[data.parents[target_id]]
    poses[:, index] = (rotations[parent].inv() * rotations[target]).as_rotvec()
  translations = np.empty((n, 3), dtype=np.float32)
  source_z_offset = 0.0
  with torch.no_grad():
    for start in range(0, n, batch_size):
      end = min(start + batch_size, n)
      count = end - start
      pose = torch.from_numpy(poses[start:end]).to(device)
      zero3 = torch.zeros(count, 3, device=device)
      result = body_model(
        betas=scope["betas"]["smplx"].to(device).expand(count, -1),
        global_orient=pose[:, 0],
        body_pose=pose[:, 1:22].reshape(count, -1),
        left_hand_pose=torch.zeros(count, 45, device=device),
        right_hand_pose=torch.zeros(count, 45, device=device),
        jaw_pose=zero3,
        leye_pose=zero3,
        reye_pose=zero3,
        expression=torch.zeros(count, body_model.num_expression_coeffs, device=device),
        transl=zero3,
        return_verts=start == 0,
      )
      source_centroid = torch.from_numpy(global_pos[start:end]).to(device).mean(dim=1)
      translations[start:end] = (
        (source_centroid - result.joints[:, :22].mean(dim=1)).cpu().numpy()
      )
      if start == 0:
        # Match the reviewed UMR LAFAN sample's initial surface-ground origin.
        # One constant shift preserves later jumps and all relative motion.
        source_z_offset = -float(
          result.vertices[0, :, 2].min().cpu() + translations[0, 2]
        )
  translations[:, 2] += source_z_offset
  return poses, translations, source_z_offset


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--upstream",
    type=Path,
    default=Path("/home/zirui/下载/UMR/lafan_to_smplx/lafan_to_smplx.py"),
  )
  parser.add_argument(
    "--vendor-dir",
    type=Path,
    default=Path(
      "/home/zirui/下载/gmr-dev/general_motion_retargeting/utils/lafan_vendor"
    ),
  )
  parser.add_argument(
    "--model-dir", type=Path, default=Path("/home/zirui/下载/models_smplx_v1_1/models")
  )
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--motion", default="")
  parser.add_argument(
    "--reference-smplx",
    type=Path,
    default=Path("/home/zirui/下载/UMR/sample_data/lafan1_smplx/dance1_subject2.npz"),
  )
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError("Run in the uv CUDA environment with GPU access")
  torch.set_num_threads(2)
  device = torch.device("cuda:0")
  scope, extract, utils = load_conversion_sources(args.upstream, args.vendor_dir)
  body_model = (
    smplx.create(
      model_path=str(args.model_dir),
      model_type="smplx",
      gender="neutral",
      ext="npz",
      use_pca=False,
      num_betas=16,
      betas=scope["betas"]["smplx"].clone(),
    )
    .to(device)
    .eval()
  )
  records = json.loads(args.manifest.read_text())
  if args.motion:
    records = [r for r in records if r["motion"] == args.motion]
  args.output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = args.output_dir.parent / "smplx_conversion_summary.json"
  summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
  upstream_hash = hashlib.sha256(args.upstream.read_bytes()).hexdigest()
  for index, record in enumerate(records, 1):
    name, source = record["motion"], Path(record["bvh"])
    target = args.output_dir / f"{name}.npz"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != record["sha256"]:
      raise ValueError(f"Source changed since selection: {source}")
    if target.exists() and summary.get(name, {}).get("status") == "complete":
      print(f"SKIP {index}/{len(records)} {name}", flush=True)
      continue
    started = time.monotonic()
    poses, translations, source_z_offset = convert(
      source, body_model, scope, extract, utils, args.batch_size, device
    )
    assert (
      len(poses) == record["frames"]
      and np.isfinite(poses).all()
      and np.isfinite(translations).all()
    )
    reference_check = None
    if args.reference_smplx.exists() and name == args.reference_smplx.stem:
      with np.load(args.reference_smplx) as reference:
        angular_error = (
          R.from_rotvec(reference["poses"].reshape(-1, 3)).inv()
          * R.from_rotvec(poses.reshape(-1, 3))
        ).magnitude()
        translation_error = np.linalg.norm(reference["trans"] - translations, axis=1)
        reference_check = {
          "max_rotation_error_rad": float(angular_error.max()),
          "max_translation_error_m": float(translation_error.max()),
          "translation_delta_mean_m": (translations - reference["trans"])
          .mean(axis=0)
          .tolist(),
          "translation_delta_std_m": (translations - reference["trans"])
          .std(axis=0)
          .tolist(),
        }
        (args.output_dir.parent / "reference_comparison.json").write_text(
          json.dumps(reference_check, indent=2)
        )
        if angular_error.max() > 3e-5 or translation_error.max() > 3e-4:
          raise ValueError(
            f"Conversion differs from the reviewed example: {reference_check}"
          )
    np.savez_compressed(
      target,
      poses=poses,
      trans=translations,
      gender="neutral",
      mocap_frame_rate=30,
      betas=scope["betas"]["smplx"].numpy().reshape(-1),
      output_up="z",
      source_bvh=str(source),
      source_sha256=digest,
      converter_sha256=upstream_hash,
      source_ground_align="first_frame_surface",
      source_z_offset_m=np.array(source_z_offset),
    )
    summary[name] = {
      "status": "complete",
      "frames": len(poses),
      "fps": 30,
      "file": str(target),
      "source_sha256": digest,
      "converter_sha256": upstream_hash,
      "seconds": time.monotonic() - started,
      "reference_check": reference_check,
      "source_z_offset_m": source_z_offset,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
      f"CONVERT {index}/{len(records)} {name}: {len(poses)} frames, {time.monotonic() - started:.1f}s, reference={reference_check}",
      flush=True,
    )


if __name__ == "__main__":
  main()
