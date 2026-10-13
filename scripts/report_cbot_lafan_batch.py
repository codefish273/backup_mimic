"""Write the final CBot LAFAN batch report and comparison figure."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, default=Path("outputs/umr_lafan40"))
  parser.add_argument(
    "--baseline", type=Path, default=Path("outputs/lafan_audit_2026-09-10")
  )
  args = parser.parse_args()
  root = args.root.resolve()
  sources = json.loads((root / "source_manifest.json").read_text())
  checks = json.loads((root / "motion_checks.json").read_text())
  output = json.loads((root / "output_manifest.json").read_text())
  if len(sources) != 40 or len(checks) != 40 or len(output) != 40:
    raise ValueError("All 40 motions must be complete before reporting")
  if any(c["status"] != "checked" for c in checks.values()):
    raise ValueError("Invalid results remain")
  names = [s["motion"] for s in sources]
  old = {
    r["motion"]: r for r in json.loads((args.baseline / "contacts.json").read_text())
  }
  minima = {label: [] for label in ("G1", "R2R CBot", "UMR CBot")}
  totals = {
    label: Counter() for label in ("G1", "R2R CBot", "UMR CBot", "UMR expanded proxy")
  }
  medians = {label: [] for label in minima}
  for name in names:
    for label, model in (("G1", "g1_training"), ("R2R CBot", "cbot_training")):
      with np.load(args.baseline / "frames" / f"{name}_{model}.npz") as data:
        z = data["foot_bottom_m"].min(axis=1)
      minima[label].append(z)
      medians[label].append(float(np.median(z)))
      metrics = old[name][model]
      for field in (
        "ground_contact_fraction",
        "nonfoot_ground_contact_fraction",
        "self_contact_fraction",
      ):
        totals[label][field] += metrics[field] * old[name]["frames"]
    with np.load(root / "checks/frames" / f"{name}_cbot_training.npz") as data:
      z = data["foot_bottom_m"].min(axis=1)
    minima["UMR CBot"].append(z)
    medians["UMR CBot"].append(float(np.median(z)))
    for label, model in (
      ("UMR CBot", "cbot_training"),
      ("UMR expanded proxy", "cbot_expanded_proxy"),
    ):
      for field in (
        "ground_contact_fraction",
        "nonfoot_ground_contact_fraction",
        "self_contact_fraction",
      ):
        totals[label][field] += checks[name][model][field] * checks[name]["frames"]
  n = sum(s["frames"] for s in sources)
  summary = {}
  for label in minima:
    values = np.concatenate(minima[label])
    assert len(values) == n
    summary[label] = {
      "foot_height_median_m": float(np.median(values)),
      "foot_height_p95_m": float(np.quantile(values, 0.95)),
      "both_feet_above_2cm_fraction": float(np.mean(values > 0.02)),
      **{key: float(value / n) for key, value in totals[label].items()},
    }
  summary["UMR expanded proxy"] = {
    key: float(value / n) for key, value in totals["UMR expanded proxy"].items()
  }
  flag_counts = Counter(flag for record in checks.values() for flag in record["flags"])
  aggregate = {
    "motions": 40,
    "source_frames": n,
    "windows": sum(r["windows"] for r in output),
    "models": summary,
    "flag_counts": dict(flag_counts),
  }
  (root / "comparison_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
  fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
  colors = {"G1": "#6b7280", "R2R CBot": "#d97706", "UMR CBot": "#2563eb"}
  x = np.arange(len(names))
  for label in minima:
    axes[0].plot(
      x,
      np.array(medians[label]) * 1000,
      ".-",
      label=label,
      color=colors[label],
      markersize=5,
    )
    values = np.sort(np.concatenate(minima[label]))
    sample = np.linspace(0, len(values) - 1, 2500).astype(int)
    axes[1].plot(
      values[sample] * 1000,
      (sample + 1) / len(values) * 100,
      label=label,
      color=colors[label],
    )
  axes[0].set_xticks(x, names, rotation=65, ha="right", fontsize=7)
  axes[0].set_ylabel("Median minimum foot clearance (mm)")
  axes[0].grid(alpha=0.2)
  axes[0].legend()
  axes[1].set_xlim(-30, 150)
  axes[1].set_ylim(0, 100)
  axes[1].axvline(20, color="black", alpha=0.3, linestyle="--")
  axes[1].set_xlabel("Minimum of left/right foot capsule clearance (mm)")
  axes[1].set_ylabel("Frames at or below clearance (%)")
  axes[1].grid(alpha=0.2)
  axes[1].legend()
  fig.suptitle(
    f"LAFAN: all 40 clips, {n:,} aligned source frames (kinematic geometry audit)"
  )
  fig.savefig(root / "foot_clearance_comparison.png", dpi=150)
  plt.close(fig)
  lines = [
    "**Cbot LAFAN 全量重定向与检查结果**",
    "",
    f"已完成当前 `datasets/csv_cbot/lafan/` 中 40 个 CSV 文件名对应的动作，共 {n:,} 个源帧。逐一匹配 `/home/zirui/下载/gmr-dev/lafan1` 中的同名 BVH。",
    "流程为 BVH → 人体 SMPL-X → UMR → Cbot，全程没有使用 G1 机器人动作作为中间表示。",
    f"每条原始 30 FPS 序列均保留全部帧；另生成 50 FPS SMP 运动窗口，共 {aggregate['windows']:,} 个 `(10,59)` 窗口。",
    "",
    "UV 环境位于 `/home/zirui/下载/UMR/.venv`，Python 3.12、PyTorch 2.4.1+cu121；RTX 3060 GPU 运算已验证。",
    "BVH 转换复用了本地 lafan_to_smplx 的旋转映射及固定 beta，使用批量 SMPL-X 计算。源序列仅做一次首帧人体表面贴地平移，后续腾空和相对运动保留。",
    "与已人工检查的 dance1_subject2 人体示例相比，旋转一致，根平移最大误差约 1 微米。",
    "",
    "**产物**",
    "",
    f"- [选定源文件及 SHA256]({root}/source_manifest.json)",
    f"- [Cbot CSV → 同名 BVH 逐项映射]({root}/cbot_csv_to_bvh_mapping.csv)",
    f"- [UMR 原始 qpos 序列]({root}/retarget/cbot_v3_pelvis_flatskin)",
    f"- [命名关节 CSV]({root}/csv)",
    f"- [SMP 训练窗口]({root}/smp_npz)",
    f"- [新归一化统计]({root}/norm_stats.npz)",
    f"- [逐条检查 CSV]({root}/motion_checks.csv)",
    f"- [逐帧接触数据]({root}/checks/frames)",
    f"- [输出文件与 SHA256]({root}/output_manifest.json)",
    f"- [配置、代码及运行环境来源]({root}/provenance.json)",
    "",
    "**全量几何对比**",
    "",
    "以下是逐帧摆放参考姿态的几何接触检查，不是物理 rollout 的接触率。跳跃和跑步有正常腾空帧；G1 数据也不是无碰撞真值。",
    "",
    "| 模型/动作来源 | 最低脚底高度中位数 | 双脚高于 2 cm | 任意触地 | 非脚部触地 | 自接触 |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
  ]
  for label in minima:
    row = summary[label]
    lines.append(
      f"| {label} | {row['foot_height_median_m'] * 1000:.2f} mm | {row['both_feet_above_2cm_fraction']:.2%} | {row['ground_contact_fraction']:.2%} | {row['nonfoot_ground_contact_fraction']:.2%} | {row['self_contact_fraction']:.2%} |"
    )
  lines += [
    "",
    f"![足底高度对比]({root}/foot_clearance_comparison.png)",
    "",
    "Cbot 当前训练模型的多个 `_zone` / `_safety_zone` 碰撞体被命名过滤关闭。额外检查启用了这些近似几何，但它们包含腰部相邻结构的覆盖区，不能直接全部当作训练终止条件。",
    f"UMR 在当前训练碰撞模型中的自接触率为 {summary['UMR CBot']['self_contact_fraction']:.2%}，在扩展近似模型中为 {summary['UMR expanded proxy']['self_contact_fraction']:.2%}。这些数字必须结合具体刚体对、接触深度和动作语义解释。",
    "",
    "**检查标记**",
    "",
    "所有文件通过有限数值、关节名称、四元数、源文件追溯、帧率与帧数检查。下列标记是进一步复核线索，不代表整条动作无效。跌倒起身的身体触地、手部互相接触等可能是动作本身的一部分。",
    "",
    "| 标记 | 动作数 |",
    "| --- | ---: |",
  ]
  lines.extend(f"| `{flag}` | {count} |" for flag, count in flag_counts.most_common())
  review_cases = []
  for name in names:
    for flag, model, field, threshold, label in (
      (
        "ground_penetration_over_1cm",
        "cbot_training",
        "ground_penetration_m",
        0.01,
        "训练碰撞体穿地",
      ),
      (
        "nontrunk_proxy_penetration_over_2cm",
        "cbot_expanded_proxy",
        "self_nontrunk_penetration_m",
        0.02,
        "含四肢的近似几何穿透",
      ),
    ):
      if flag not in checks[name]["flags"]:
        continue
      with np.load(root / "checks/frames" / f"{name}_{model}.npz") as data:
        values = data[field]
      frame = int(values.argmax())
      review_cases.append(
        {
          "motion": name,
          "flag": flag,
          "kind": label,
          "frame_zero_based": frame,
          "time_s": frame / 30,
          "max_depth_m": float(values[frame]),
          "frames_over_threshold": int(np.sum(values > threshold)),
          "total_frames": len(values),
          "result": checks[name]["file"],
        }
      )
  (root / "review_queue.json").write_text(
    json.dumps(review_cases, indent=2, ensure_ascii=False) + "\n"
  )
  if review_cases:
    lines += [
      "",
      "优先复核的位置（帧号从 0 开始）：",
      "",
      "| 动作 | 类型 | 最大处帧号 | 时间 s | 深度 |",
      "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
      f"| {item['motion']} | {item['kind']} | {item['frame_zero_based']} | {item['time_s']:.2f} | {item['max_depth_m'] * 1000:.1f} mm |"
      for item in review_cases
    )
    lines += ["", f"完整定位数据见 [review_queue.json]({root}/review_queue.json)。"]
  lines += [
    "",
    "`inspect_proxy_contact_pairs`：扩展碰撞近似中较多自接触，优先核对腰部相邻结构和几何排除；`nontrunk_proxy_penetration_over_2cm`：包含四肢的深接触需要逐帧查看；`ground_penetration_over_1cm`：训练碰撞体穿地；`foot_clearance_on_source_support`：在 G1 参考支撑时段仍有较多脚部离地；`joint_step_over_half_radian`：单帧关节变化超过 0.5 rad。具体帧、刚体对和深度保存在 checks/ 下。",
    "",
    "**批处理修复与保留项**",
    "",
    "执行中发现 UMR 批量入口忽略 JSON 中的 sequential 模式，强制使用 bidirectional，导致已检查示例出现约 1 rad 的关节跳变。已修复配置读取，验证 JSON 与 CLI 优先级后重跑。修复后该示例最大单帧变化回到约 0.359 rad，与单条流程一致。",
    f"修复前部分结果完整保留在 [before_warmstart_fix]({root}/before_warmstart_fix)。",
    "",
    "此前的 R2R 时长修复产物仍位于 datasets/npz_cbot/lafan 和 forward；UMR 新结果独立保存，未覆盖旧数据或现有先验/RL 权重。训练应使用与新数据匹配的归一化文件，并重新训练先验。尚未完成新动作的动态跟踪训练验证。",
    "",
    "**MuJoCo 查看某一条动作**",
    "",
    "```bash",
    "uv run --directory /home/zirui/下载/UMR --frozen python scripts/visualize_robot_retarget_result.py \\",
    f"  --result {root}/retarget/cbot_v3_pelvis_flatskin/walk1_subject1_cbot_v3_pelvis_flatskin.npz \\",
    "  --viewer-backend glfw-ui --play --no-show-robot-slots --no-show-ground-contact-map",
    "```",
    "",
    "空格暂停/继续，A/D 逐帧，Q/E 跳转约一秒，[ / ] 调速，进度条定位。将文件名替换为其它动作即可。",
  ]
  (root / "README.md").write_text("\n".join(lines) + "\n")
  print(root / "README.md")
  print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
  main()
