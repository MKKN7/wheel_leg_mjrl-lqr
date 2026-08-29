# Wheel-Leg Locomotion Sandbox

MuJoCo and MuJoCo-Warp training sandbox for a wheeled-leg robot. The repository
is organized by role so the simulation, training, evaluation, and test entry
points remain easy to find.

## Directory Map

- `configs/` - YAML configuration for scenes, batch physics, PPO, and curricula.
- `assets/`, `meshes/`, `meshes_mj/` - source assets and MuJoCo mesh resources.
- `artifacts/` - generated checkpoints, metrics, and previews.
- `tests/` - unit and regression tests.
- `docs/` - implementation notes and integration guides.
- `reports/` - generated asset classification reports and diagnostic logs.

## Main Entry Points

- `train_ppo.py` and `train_bc.py` - CPU residual-policy training.
- `train_warp_ppo.py` and `train_warp_curriculum.py` - CUDA/MuJoCo-Warp training.
- `train_full_curriculum.py` - complete curriculum orchestration.
- `evaluate_policy.py` and `evaluate_official_full_course.py` - evaluation.
- `view_rm_train_ground.py` and `view_wheeled_infantry.py` - interactive scene viewers.

## Testing

All project commands must use the `robot7` Python 3.12 environment. VS Code is
configured to use it for Run and Debug. From a terminal, run either the included
launcher or activate the Conda environment:

```powershell
./robot7_python.cmd train_warp_ppo.py
./run_tests_robot7.cmd

# Equivalent interactive shell workflow
conda activate robot7
python -m unittest discover -s tests -t .
```

Do not launch project scripts by double-clicking `.py` files: Windows currently
associates them with the system Python 3.14. The required MuJoCo and learning
dependencies are installed in `robot7` (Python 3.12).

GPU-backed checks remain opt-in where the individual tests document an
environment-variable gate.
