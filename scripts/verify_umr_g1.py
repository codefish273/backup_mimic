"""Verify completed G1 artifacts and produce the offline audit handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.csv_to_npz import EE_BODY_NAMES, JOINT_NAMES
from scripts.prepare_umr_g1 import digest, write_json

from smp.pretrain.dataset import MotionWindowDataset


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path("outputs/umr_g1_lafan40"))
  args = parser.parse_args()
  root = args.root
  manifest = json.loads((root / "dataset_manifest.json").read_text())
  records = manifest["records"]
  names = {record["motion"] for record in records}
  expected_run = {"run1_subject2", "run1_subject5", "run2_subject1", "run2_subject4"}
  assert len(names) == len(records) == 40
  assert {path.stem for path in (root / "smp_npz").glob("*.npz")} == names
  assert {path.stem for path in (root / "run_npz").glob("*.npz")} == expected_run
  hashes = {}
  review = []
  for record in records:
    name = record["motion"]
    path = root / "smp_npz" / f"{name}.npz"
    hashes[str(path.relative_to(root))] = digest(path)
    assert hashes[str(path.relative_to(root))] == record["output_sha256"]
    with np.load(path, allow_pickle=False) as data:
      assert data["windows"].shape == (record["windows"], 10, 59)
      assert data["windows"].dtype == np.float32
      assert np.isfinite(data["windows"]).all()
      assert float(data["fps"].item()) == 50
      assert data["joint_names"].tolist() == list(JOINT_NAMES)
      assert data["ee_body_names"].tolist() == list(EE_BODY_NAMES)
      assert int(data["input_frame_count"]) == record["source_frames"]
      assert int(data["resampled_frame_count"]) - 9 == record["windows"]
    raw = root / "retarget/unitree_g1" / f"{name}_unitree_g1.npz"
    assert digest(raw) == record["provenance"]["result_sha256"]
    for rate, label in ((30, "source_30hz"), (50, "training_50hz")):
      frame_path = root / "checks/frames" / f"{name}_{rate}hz.npz"
      with np.load(frame_path, allow_pickle=False) as frames:
        for field in ("ground_penetration_m", "self_penetration_m"):
          depths = frames[field]
          assert np.isfinite(depths).all()
          if depths.max() > 0.01:
            worst = int(depths.argmax())
            review.append(
              {
                "motion": name,
                "fps": rate,
                "kind": field,
                "frame_zero_based": worst,
                "time_s": worst / rate,
                "peak_depth_m": float(depths[worst]),
                "frames_over_1cm": int(np.sum(depths > 0.01)),
                "fraction_over_1cm": record[label][field]["fraction_over_1cm"],
                "result": str(raw),
                "arrays": str(frame_path),
              }
            )
    if name in expected_run:
      run_path = root / "run_npz" / path.name
      assert digest(run_path) == digest(path)
      hashes[str(run_path.relative_to(root))] = digest(run_path)
  stats_path = root / "norm_stats.npz"
  with np.load(stats_path, allow_pickle=False) as stats:
    assert stats["q_low"].shape == stats["q_high"].shape == (59,)
    assert np.isfinite(stats["q_low"]).all() and np.isfinite(stats["q_high"]).all()
    assert np.all(stats["q_high"] > stats["q_low"])
  hashes["norm_stats.npz"] = digest(stats_path)
  dataset = MotionWindowDataset(root / "run_npz", stats_path)
  assert len(dataset) == 48224 and torch.isfinite(dataset.windows).all()
  assert sum(record["windows"] for record in records) == 440760
  verification = {
    "status": "passed",
    "motions": len(records),
    "total_windows": 440760,
    "run_motions": 4,
    "run_windows": len(dataset),
    "shape": [10, 59],
    "dataset_loader_verified": True,
    "training_started": False,
    "flagged_motions": sum(bool(record["flags"]) for record in records),
    "artifact_sha256": hashes,
  }
  write_json(root / "verification.json", verification)
  write_json(root / "review_queue.json", review)
  lines = [
    "# G1 UMR LaFan 数据交付",
    "",
    "40 条同源动作已完成重定向、转换与离线审计；未运行 SMP 预训练或后训练。",
    "",
    "## 数据入口",
    "",
    "- `run_npz/`：四条 run，共 48,224 个 `(10, 59)` float32 窗口。",
    "- `smp_npz/`：完整 40 条，共 440,760 个窗口。",
    "- `norm_stats.npz`：由本次 G1 完整 40 条窗口计算的 q01/q99，不能与 cbot 统计混用。",
    "- `csv/`：30 Hz，注释声明帧率与具名关节顺序；数值列为 root xyz、xyzw、29 个关节，无 time 列。",
    "- `retarget/unitree_g1/`：原始 30 Hz UMR qpos。",
    "- `checks/`、`review_queue.json`：30/50 Hz 几何审计、接触对计数及穿透峰值帧。",
    "- `verification.json`：最终读回、SHA-256、MotionWindowDataset 加载验证。",
    "",
    "## Run 审计（50 Hz 训练碰撞模型）",
    "",
    "| 动作 | 窗口 | 自穿透峰值 mm | 深于 1 cm 帧比例 | 峰值帧 |",
    "| --- | ---: | ---: | ---: | ---: |",
  ]
  for record in records:
    if record["motion"] in expected_run:
      metrics = record["training_50hz"]["self_penetration_m"]
      lines.append(
        f"| {record['motion']} | {record['windows']} | {metrics['max'] * 1000:.2f} | "
        f"{metrics['fraction_over_1cm']:.3%} | {metrics['worst_frame']} |"
      )
  lines += [
    "",
    "全部 run 无关节越限，最大穿地不足 2 mm；自接触主要涉及手腕与髋部。",
    "run2 两条触发较多自穿透；run1_subject2 虽未触发比例阈值，仍有短暂约 47 mm 峰值。",
    "比例标记规则为深于 1 cm 的帧超过 1%；无标记不代表完全无穿透。",
    "完整 40 条中 30 条有复核标记。未删除、裁剪或修正这些动作，以保持与 cbot 的同源对照。",
    "review_queue 同时记录短暂峰值；50 Hz 帧号不能直接当作 30 Hz 原始结果的帧号。",
    "",
    "## 契约与限制",
    "",
    "使用当前 g1_smp_env_cfg 训练碰撞模型执行 CPU MuJoCo FK，复用现有插值、速度及窗口计算。",
    "UMR G1 与训练 G1 的零姿态 torso 原点存在 10 mm 差异，因此末端位置必须通过训练模型重算。",
    "这不是动态跟踪验证，也不意味着数据已满足所有训练质量要求。",
    "新增三个单元测试及 Ruff 通过；全仓 unittest 有两个既有导入错误（RobotMotionLoader、_read_csv_motion 缺失），详见 unittest.log。",
  ]
  (root / "README.md").write_text("\n".join(lines) + "\n")
  print(
    json.dumps(
      {key: value for key, value in verification.items() if key != "artifact_sha256"},
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
