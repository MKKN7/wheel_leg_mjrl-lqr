# Isaac Lab 集成技能

## 概述

Isaac Lab 基于 NVIDIA Isaac Sim，提供 GPU 并行强化学习环境。必须遵循其 `BaseEnv`、`RLTaskEnv`、`Manager` 等设计模式。

## 环境搭建

- 使用 `isaaclab` 包，环境类继承自 `RLTaskEnv` 或 `BaseEnv`。
- 在 `__init__` 中加载 `config` 字典，禁止硬编码参数。
- 场景构建使用 `InteractiveScene`，通过 `cfg` 定义机器人、地面、传感器等。
- 动作和观测维度、限位、噪声等全部在配置中定义。

## Task 设计

- 继承 `BaseTask`，实现 `reset_idx`、`get_observations`、`get_rewards`、`pre_physics_step`、`post_physics_step`。
- 使用 `EventManager` 处理域随机化（质量、摩擦、初始姿态等），避免手动设置。
- 观测管理器使用 `ObservationManager` 及其子类（如 `ObsGroup`），按组管理观测。
- 动作管理器使用 `ActionManager`，支持 PD 控制、直接力矩等，输出需经过 `clip`。

## 多环境并行

- 环境数量由 `num_envs` 配置，通常设为 GPU 可支持的数千个。
- 所有张量操作在 GPU 上完成，禁止在循环内使用 `.cpu().numpy()` 进行频繁同步。
- 数据采集使用 `env.step(actions)` 返回 PyTorch 张量，保持批次维度。
- 与训练框架（RL Games、SKRL、Stable-Baselines3）集成时，使用 `VecEnv` 包装。
- 与 RSL-RL 集成时，环境类直接传入 `RslRlOnPolicyRunner`，无需额外 `VecEnv` 包装。
- 与自定义 PPO 集成时，环境需自行管理 `num_envs` 的批量状态。

## 域随机化

- 使用 `EventManager` 注册随机化事件，配置每环境的随机化范围和概率。
- 随机化参数在 `reset` 时批量生成，避免逐环境循环。
- 观测噪声、延迟等在 Manager 中实现。

## 训练集成

### 首选：RSL-RL（足式/轮腿机器人）

- 使用 `rsl_rl` 库，通过 `RslRlOnPolicyRunner` 启动。
- 配置文件必须包含 `runner`、`policy`、`algorithm` 三部分。
- 策略类继承 `ActorCritic` 或 `ActorCriticRecurrent`。
- 支持非对称 Actor-Critic：critic 观测组可包含特权信息（如接触力、地面摩擦），actor 观测组仅限机载传感器。
- 支持 Teacher-Student 蒸馏：特权策略训练后，通过 `StudentPolicy` 蒸馏到无特权策略。
- 与自定义 PPO 切换条件：当需要非对称 AC、蒸馏、多 GPU 分布式时，必须迁移到 RSL-RL。

### 备选：自定义 PPO（迁移期/复现基准）

- 保留 Warp 阶段验证过的 PPO 核心（GAE、Clipped Loss、Value Loss）。
- 环境需实现 `step(actions: torch.Tensor) -> (obs, reward, reset, extras)`，返回 GPU 张量。
- 禁止在 rollout 循环内使用 `.cpu().numpy()`。
- 适用场景：Isaac Lab 环境初版验证、Warp-Isaac 训练曲线对比、快速调试。

### 第三方框架

- `rl_games`：通用 GPU 并行，适合标准 Gym 任务。
- `skrl`：支持多种算法（PPO、SAC、TRPO）。
- `Stable-Baselines3`：仅用于 CPU 小规模实验，不推荐 GPU 并行。
- 使用 `isaaclab_rl` 或 `rl_games` 等库进行 PPO 训练。
- 训练配置 YAML 中必须包含：算法参数、网络结构、奖励权重、域随机化参数。
- 启动训练前运行 `env` 的集成测试（如 `random_actions_test`）。

## 性能优化

- 使用 GPU 管道（pipeline）和 CUDA 图减少 CPU 开销。
- 避免在 `step` 内调用 Python 级循环或同步点。
- 监控 FPS、GPU 利用率，目标 FPS 不低于 1000（依赖环境复杂度）。
- 使用 `torch.compile` 或 `torch.jit.script` 优化奖励和观测计算。

## 常见错误排查

- **CUDA OOM**：减少 `num_envs` 或减小观测/动作维度，使用 `empty_cache` 但不应频繁调用。
- **渲染冲突**：无头模式使用 `--headless`，关闭不必要的相机传感器。
- **环境重置缓慢**：检查 `reset_idx` 是否使用向量化操作，避免逐环境重置。
- **动作未生效**：确认 ActionManager 的 `clip_actions` 已启用。

## 配置示例（YAML 结构）

```yaml
env:
  num_envs: 4096
  env_spacing: 3.0
  episode_length_s: 20.0
  device: "cuda:0"

task:
  robot: "quadruped"
  control_type: "torque"
  action_scale: 1.0
  clip_actions: true

domain_randomization:
  mass_range: [0.8, 1.2]
  friction_range: [0.5, 1.5]
  reset_prob: 0.5
  