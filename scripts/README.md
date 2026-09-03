# 脚本入口

这里的每个 `.py` 都是可直接执行的薄入口：它会从 `src/wheel_leg_mjrl/` 加载真正实现。这样根目录只保留场景和项目文件，运行命令集中在一个位置。

从仓库根目录执行，先完成根 README 的环境安装步骤：

```powershell
python scripts/lqr_deploy.py --help
```

| 入口 | 用途 |
| --- | --- |
| `lqr_deploy.py` | MuJoCo 物理模型 LQR；支持无界面验证。 |
| `control.py` | 控制相关的兼容入口。 |
| `train_ppo.py` | CPU 残差 PPO 训练。 |
| `train_bc.py` | 以 LQR 演示数据做行为克隆。 |
| `train_warp_ppo.py` | MuJoCo-Warp 平地预检与 PPO。 |
| `train_warp_curriculum.py` | capability-gated GPU 课程阶段。 |
| `train_full_curriculum.py` | 完整课程编排。 |
| `evaluate_policy.py` | 通用策略评估。 |
| `evaluate_official_full_course.py` | 官方全赛道评估，需要显式 checkpoint。 |
| `view_wheeled_infantry.py` | 交互查看基础场景。 |
| `view_rm_train_ground.py` | 交互查看 RMUC 训练场地。 |
| `render_guide_wheel_preview.py` | 重新生成导轮预览图。 |

配置文件一律放在 `configs/`。例如，显式指定平地 GPU 预检配置：

```powershell
python scripts/train_warp_ppo.py --config configs/warp_flat_batch.yaml --smoke
```

使用 `--help` 查看参数，不要将临时脚本放回仓库根目录；探索性代码应存放在 `experiments/probes/`。
