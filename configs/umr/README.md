UMR 的 Cbot LAFAN 试验使用人体 SMPL-X 作为输入，直接输出 Cbot 关节运动。
这条路径不使用 G1 运动作为中间表示。

当前 UMR 源码目录为 `/home/zirui/下载/UMR`，其 `pyproject.toml`、`uv.lock` 和 `.venv`
由 uv 管理，使用 Python 3.12 和 `torch==2.4.1+cu121`。GPU 运算已在 RTX 3060 上验证。

```bash
cd /home/zirui/下载/UMR
uv sync --locked
uv run --frozen python scripts/humanoid_retarget_pipeline.py \
  --config /home/zirui/mimic_clai/configs/umr/cbot_lafan_full.json \
  --stage retarget --skip-view
```

`cbot_lafan.json` 运行前 300 帧；`cbot_lafan_full.json` 运行完整 3945 帧。
首次运行会为 Cbot 与当前人体形状建立并训练 4096 点对应关系，后续复用缓存。
SMPL-X 批量为 64 帧，以适应当前 6 GB 显存。Cbot 已有 floating joint，配置关闭了 UMR
额外生成 floating-root XML 的步骤。

原始 UMR 输出为 `qpos` NPZ，可通过以下路径转换为 SMP 需要的 59 维运动窗口：

```bash
cd /home/zirui/mimic_clai
uv run scripts/umr_cbot_to_csv.py \
  --input-file outputs/umr_cbot/dance1_subject2_full.npz \
  --output-file outputs/umr_cbot/csv_full/dance1_subject2.csv
uv run scripts/csv_to_npz.py --robot cbot \
  --input-dir outputs/umr_cbot/csv_full \
  --output-dir outputs/umr_cbot/smp_npz_full \
  --output-fps 50 --fk-backend mujoco
```

新导出的 CSV 有正确的 30 FPS 时间戳，读取时无需覆盖输入帧率。
试验文件位于 `outputs/umr_cbot/`，不混入已有 R2R 数据集。
结果验证覆盖数组、时间基准、关节顺序和离线几何接触；尚未完成动态跟踪训练验证。

本次输入是 UMR 随附的 `sample_data/lafan1_smplx/dance1_subject2.npz`。
它已经是从 LAFAN 转出的 SMPL-X。将其它原始 LAFAN BVH 纳入此路径时，需要先用
UMR 推荐的 `lafan_to_smplx` 或同等人体转换流程转换为 SMPL-X。

40 条批处理使用 `/home/zirui/下载/gmr-dev/lafan1` 中的原始 BVH；名称列表为
`configs/umr/lafan40_motion_names.txt`，含帧数和源文件 SHA256 的选择清单保存在
`outputs/umr_lafan40/source_manifest.json`。批量转换适配器复用本地
`UMR/lafan_to_smplx` 的映射和 beta，以及 GMR 的 BVH 解析/FK 工具；不执行 G1 重定向。
一次性的首帧人体表面贴地处理与已检查的 UMR 示例一致，后续相对运动和腾空保留。

```bash
uv run python -m scripts.select_cbot_lafan_sources \
  --cbot-csv-dir datasets/csv_cbot/lafan \
  --bvh-dir /home/zirui/下载/gmr-dev/lafan1
uv run --directory /home/zirui/下载/UMR --frozen python \
  /home/zirui/mimic_clai/scripts/lafan_bvh_to_smplx.py \
  --manifest /home/zirui/mimic_clai/outputs/umr_lafan40/source_manifest.json \
  --output-dir /home/zirui/mimic_clai/outputs/umr_lafan40/smplx
uv run --directory /home/zirui/下载/UMR --frozen python \
  /home/zirui/mimic_clai/scripts/run_cbot_lafan_batch.py
```

40 条名单只由当前 Cbot CSV 文件名决定，G1 CSV 不参与源文件选择。
逐项映射保存在 `outputs/umr_lafan40/cbot_csv_to_bvh_mapping.csv`。

批流程使用 3 个处理进程、CUDA 0、64 帧 SMPL-X 批次、300 帧数据块和 sequential
warm start，并复用已训练的对应关系。`run_status.json` 记录进度；每条输出检查后导出
CSV，全部完成后再生成 50 FPS SMP NPZ、配套归一化和输出清单。

当前 UMR 批量入口存在忽略 JSON warm-start 模式的上游问题。本次已应用
`respect_warmstart_config.patch`，使配置生效；重新检出 UMR 时需要保留此修复。
修复前的部分双向结果保存在 `outputs/umr_lafan40/before_warmstart_fix`，不作为最终数据。

检查覆盖帧数/帧率、四元数与关节顺序、关节限制和单帧变化、脚底高度、触地与自接触。
扩展碰撞近似的腰部相邻结构接触会单独复核，检查标记不等于整段动作无效。
