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

Run the test suite from the project root:

```powershell
python -m unittest discover -s tests -t .
```

GPU-backed checks remain opt-in where the individual tests document an
environment-variable gate.
