# 实验管理技能

## 实验追踪

- 每次训练使用独立实验 ID：`YYYYMMDD_HHMMSS_算法名_环境名`。
- 所有配置、日志、权重、指标归档到 `experiments/<id>/`。
- 使用 TensorBoard 或 WandB 记录训练曲线、奖励分解、Q 值分布、动作分布、域随机化参数分布。
- Isaac Lab 实验中额外记录 GPU 利用率、FPS、并行环境数。

## 超参数管理

- 所有超参数写入 YAML 配置文件，禁止硬编码。
- 支持命令行覆盖配置，便于批量实验。
- Isaac Lab 与 MuJoCo Warp 的配置可合并到同一 YAML 或分文件管理。

## 结果记录

- 保存最终策略的 ONNX 或 TorchScript 模型，附环境版本、依赖列表、训练配置快照。
- 记录评估指标（平均回报、任务成功率、鲁棒性测试结果）并生成报告。
- 记录 Isaac Lab 与 MuJoCo Warp 版本、CUDA 版本、GPU 型号等环境信息。

## 版本控制

- 训练代码、环境代码、配置文件纳入 Git 管理。
- 实验产出物使用 DVC 或 Git LFS 管理。
- 禁止修改历史实验配置或结果，新实验创建新目录。

## 可视化

- 定期渲染评估视频，保存到实验目录。
- 绘制域随机化参数与性能的敏感性曲线。
- Isaac Lab 可使用 `Viewer` 录制视频。

## 目录模板

参考 `.codex/templates/experiment_structure.md`（若未提供，可按以下结构创建）：
experiments/
└── YYYYMMDD_HHMMSS_算法名_环境名/
├── config.yaml           # 完整配置快照
├── logs/                 # TensorBoard / WandB 日志
├── models/               # 策略权重（ONNX/TorchScript）
├── metrics/              # 评估指标 JSON/CSV
├── videos/               # 评估视频
└── report.md             # 实验报告
