
# MuJoCo Warp 集成技能

## 概述

Warp 是 NVIDIA 的 GPU 并行编程库，可用于加速 MuJoCo 仿真中的计算密集型部分（如接触检测、自定义约束、多环境回滚）或作为独立 GPU 物理后端。本技能指导如何安全高效地融合两者。

## 集成模式

1. **GPU 数据管道**：将 MuJoCo 输入/输出数据（状态、控制）转换为 Warp 数组，在 GPU 上并行处理，再传回 MuJoCo。
2. **自定义核函数**：使用 `wp.launch` 编写核函数，替代 MuJoCo 中高耗时的 Python 循环（如多环境 `step`）。
3. **Warp 物理后端**（若适用）：使用 `wp.sim` 构建独立物理引擎，但需保证与 MuJoCo 模型一致性。

## 核心规则

- 禁止在核函数内创建 Warp 数组或分配内存。所有 Warp 数组在初始化阶段创建，并复用。
- 使用 `wp.zeros`、`wp.zeros_like` 等创建 GPU 数组，通过 `wp.to_torch`/`wp.from_torch` 与 PyTorch 张量无缝互操作。
- 核函数必须无副作用（纯函数风格），避免竞争条件。
- 数据同步采用异步方式，避免阻塞 CPU；仅在必须读取结果时使用 `.numpy()` 或 `wp.synchronize()`。
- 检查数组索引越界，使用 `wp.atomic_add` 时注意原子性需求。

## 性能优化

- 使用 `wp.launch(kernel, dim=grid_size, inputs=[...], outputs=[...])` 启动核函数。
- 内存布局对齐：确保 Warp 数组为连续内存，避免 strides。
- 合并小核函数，减少 kernel launch 开销。
- 使用 `wp.is_math_enabled()` 检查数学功能可用性。

## 与 MuJoCo 数据交互

- 若需要将 MuJoCo 状态传入 Warp，先将 `data.qpos`、`data.qvel` 复制到 GPU 数组（使用 `wp.from_numpy` 或 PyTorch 中转）。
- 处理多环境时，将 MuJoCo 模型复制 N 份（注意内存），或使用 MJX（MuJoCo XLA）替代原生 MuJoCo 以获得 GPU 并行。
- 若使用 MJX，参考 JAX 接口，Warp 可用于辅助数据预处理或后处理。

## 测试与调试

- 单元测试中对比 Warp 核函数结果与 CPU/NumPy 计算结果，误差小于 1e-6。
- 检查内存泄漏：循环运行核函数并监控 GPU 内存占用。
- 使用 Nsight Compute 分析核函数性能。

## 配置示例（YAML 结构）

```yaml
warp:
  device: "cuda:0"
  num_envs: 1024
  kernel_block_size: 256
  enable_math: true
  