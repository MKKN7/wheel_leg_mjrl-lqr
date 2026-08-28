# 奖励设计技能

## 通用结构

奖励函数分解为可独立调试的项：

```python
reward = sum(weight_i * term_i)
每个 term_i 应归一化到 [0,1] 或 [-1,1]，保证数量级一致。

常用正向项模板

· 速度跟踪：exp(-((v - v_cmd) / scale)^2)
· 姿态跟踪：exp(-dot(ori_error, ori_error) * k)
· 高度/离地奖励：clip((h - h_min) / (h_max - h_min), 0, 1)
· 接触/着地奖励：+bonus if contact else -penalty
· 峰值增量奖励（跳跃）：记录历史峰值，只奖励新超出部分。

常用负向项模板

· 能耗：mean((tau / tau_max)^2)
· 动作变化率：mean((a_t - a_{t-1})^2)
· 关节限位接近：max(0, (q - q_limit) / margin)^2
· 横向漂移：clip(abs(v_y) / v_max, 0, 1)
· 腿长差/对称性：abs(l_left - l_right) / nominal_length

事件奖励/惩罚

· 任务成功：一次性大额正奖励（如 +10）。
· 任务失败/摔倒：一次性大额负奖励（如 -10）。
· 不安全状态：额外惩罚（如 -30），明显高于常规负向项。

稀疏性与课程

· 如果任务稀疏，使用势能函数或分阶段奖励引导。
· 训练初期可降低难项权重，随进度调整。

调试要求（MuJoCo 原生 / 通用）

· 训练日志中单独记录每个奖励项的平均值和标准差。
· 检查奖励尺度是否导致价值函数发散，必要时进行 reward scaling。尽可能减少reward hacking发生。

Isaac Lab / Torch 向量化奖励实现

计算模式

· 奖励函数必须为纯 PyTorch 张量操作，输入为批处理数据，形状为 (num_envs, ...)。
· 禁止在奖励计算中使用 Python for 循环逐环境计算。
· 每个奖励项实现为独立函数，返回形状 (num_envs,) 或 (num_envs, 1)。
· 所有中间变量保持在 GPU 上，避免 .cpu().numpy() 同步。

与 Isaac Lab 集成

· 使用 RewardManager 注册各奖励项，配置中定义函数名、权重和参数。
· 奖励函数签名示例：def _reward_velocity(self, env_ids: torch.Tensor) -> torch.Tensor。
· 在 BaseTask 的 _get_rewards 中调用管理器，禁止手动拼接字典。
· 奖励项配置示例（YAML）：

```yaml
reward:
  velocity_tracking:
    func: _reward_velocity
    weight: 1.0
    params: {scale: 0.5}
  energy_penalty:
    func: _reward_energy
    weight: -0.01
    params: {max_torque: 20.0}
```

向量化常用模板

· 速度跟踪：

```python
torch.exp(-((v_x - v_cmd) / scale) ** 2)
```

· 能耗：

```python
torch.mean((tau / tau_max) ** 2, dim=1)
```

· 动作变化率：

```python
torch.sum((a_t - a_prev) ** 2, dim=1)
```

· 关节限位接近：

```python
torch.clamp((q - q_limit_low) / margin, min=0) + torch.clamp((q_limit_high - q) / margin, min=0)
```

函数签名例子
def vel_tracking_reward(v_actual, v_target, scale=0.5) -> float:
    return math.exp(-((v_actual - v_target) / scale) ** 2)

调试与数值安全

· 在日志中记录每个奖励项在 batch 上的均值和标准差，检测 NaN/Inf。
· 使用 torch.nan_to_num 或 torch.clamp 防止异常值传播，但应先定位来源。
· 奖励总和不做隐式归一化，若 magnitude 过大，调整权重或缩放系数。
· 使用 torch.compile 或 torch.jit.script 优化复杂奖励计算（可选）。
重要：事件奖励只在终止步或特定触发帧给一次
不要在每一步都给，否则会变成密集奖励导致 value 爆炸

MuJoCo Warp 奖励计算

· 若奖励计算放在 Warp 核函数中，同样必须向量化，禁止核函数内标量循环。
· 核函数输入输出为 Warp 数组，返回前使用 wp.clamp 保证范围。
· 推荐在 PyTorch 中计算奖励，Warp 仅用于物理加速，避免重复开发。
