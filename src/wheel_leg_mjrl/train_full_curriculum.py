"""Run the YAML-declared CUDA PPO curriculum in fail-closed order."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from entrypoint_paths import project_path, resolve_cli_input
from evaluate_official_full_course import (
    _atomic_json,
    evaluate_full_course,
    load_full_evaluation_config,
)
from train_warp_curriculum import (
    WarpCurriculumConfigError,
    _stage_artifact_path,
    load_curriculum_config,
    load_residual_checkpoint,
    run_curriculum_training,
)


DEFAULT_CURRICULUM_CONFIG = project_path("configs", "warp_curriculum_ppo.yaml")
DEFAULT_FULL_EVALUATION_CONFIG = project_path("configs", "official_full_evaluation.yaml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=resolve_cli_input, default=DEFAULT_CURRICULUM_CONFIG)
    parser.add_argument(
        "--init-residual-checkpoint",
        type=resolve_cli_input,
        default=None,
        help="Optional audited CPU residual checkpoint for the first executed CUDA stage.",
    )
    parser.add_argument(
        "--from-stage",
        default=None,
        help="Resume at this manifest stage; all earlier stages must already be certified.",
    )
    parser.add_argument(
        "--full-evaluation-config",
        type=resolve_cli_input,
        default=DEFAULT_FULL_EVALUATION_CONFIG,
        help="YAML plan for fail-closed final official full-course certification.",
    )
    return parser.parse_args(argv)


def run_full_curriculum(
    curriculum_path: str | Path,
    *,
    init_residual_checkpoint: str | Path | None = None,
    from_stage: str | None = None,
    full_evaluation_config: str | Path = DEFAULT_FULL_EVALUATION_CONFIG,
) -> tuple[Path, ...]:
    """Train every YAML-declared CUDA curriculum stage in manifest order.

    ``run_curriculum_training`` owns the physical gate, candidate checkpoint,
    route evaluation, and predecessor certificate checks.  This wrapper only
    supplies deterministic ordering and refuses to continue after any error.
    """

    config = load_curriculum_config(resolve_cli_input(curriculum_path))
    full_evaluation = load_full_evaluation_config(resolve_cli_input(full_evaluation_config))
    stages = config.stages
    if not stages:
        raise WarpCurriculumConfigError("curriculum contains no CUDA course stages")
    if from_stage is not None:
        ids = tuple(stage.stage_id for stage in stages)
        if from_stage not in ids:
            raise WarpCurriculumConfigError(f"--from-stage must identify a manifest stage: {from_stage}")
        start_index = ids.index(from_stage)
        for stage in stages[:start_index]:
            artifact = _stage_artifact_path(config.output.checkpoint_path, stage)
            if not artifact.is_file():
                raise WarpCurriculumConfigError(
                    f"cannot resume at {from_stage!r}; certified predecessor artifact is missing for {stage.stage_id!r}"
                )
            load_residual_checkpoint(artifact, config=config, stage=stage)
        stages = stages[start_index:]
    outputs: list[Path] = []
    output_by_stage: dict[str, Path] = {}
    first_init = init_residual_checkpoint
    for stage in stages:
        output = run_curriculum_training(
            config,
            stage_id=stage.stage_id,
            smoke=False,
            init_residual_checkpoint=first_init,
        )
        outputs.append(output)
        output_by_stage[stage.stage_id] = output
        first_init = None

    # The declared full-course certificate belongs to the final official
    # policy, not to an intermediate route checkpoint.  Partial sequences
    # therefore remain usable for recovery but cannot emit a final certificate.
    if full_evaluation.checkpoint_stage_id in output_by_stage:
        if full_evaluation.curriculum_config_path != config.source_path:
            raise WarpCurriculumConfigError(
                "full-course evaluation YAML references a different curriculum manifest"
            )
        checkpoint_path = output_by_stage[full_evaluation.checkpoint_stage_id]
        report = evaluate_full_course(
            plan=full_evaluation,
            checkpoint_path=checkpoint_path,
        )
        _atomic_json(full_evaluation.report_path, report)
        if report.get("passed") is not True:
            raise WarpCurriculumConfigError(
                "official full-course certification failed; "
                f"report={full_evaluation.report_path}"
            )
    return tuple(outputs)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        outputs = run_full_curriculum(
            args.curriculum,
            init_residual_checkpoint=args.init_residual_checkpoint,
            from_stage=args.from_stage,
            full_evaluation_config=args.full_evaluation_config,
        )
    except (WarpCurriculumConfigError, RuntimeError, ValueError, OSError) as error:
        raise SystemExit(f"MuJoCo-Warp full curriculum blocked/failed: {error}") from error
    for output in outputs:
        print(f"certified curriculum checkpoint: {output}")


if __name__ == "__main__":
    main()
