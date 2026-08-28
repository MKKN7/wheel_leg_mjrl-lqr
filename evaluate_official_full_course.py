"""Fail-closed all-route certification for a trained official CUDA policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from official_course_warp import (
    OfficialCourseAdapterError,
    evaluate_policy_stage,
    load_official_course_config,
)
from terrain_curriculum import load_terrain_curriculum
from train_ppo import ActorCritic
from train_warp_curriculum import (
    CurriculumConfig,
    StageConfig,
    WarpCurriculumConfigError,
    _bundle_value,
    _external_stage_bundle,
    load_curriculum_config,
    validate_checkpoint_metadata,
)


FULL_EVALUATION_SCHEMA = 1


class FullCourseEvaluationError(ValueError):
    """Raised when an all-route certificate cannot be proven."""


@dataclass(frozen=True)
class FullEvaluationConfig:
    source_path: Path
    curriculum_config_path: Path
    checkpoint_stage_id: str
    full_terrain_stage_id: str
    official_stage_ids: tuple[str, ...]
    report_path: Path


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FullCourseEvaluationError(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise FullCourseEvaluationError(f"{name} must be a non-empty string")
    return value


def _path(source: Path, value: Any, name: str, *, must_exist: bool) -> Path:
    candidate = Path(_string(value, name))
    result = candidate.resolve() if candidate.is_absolute() else (source.parent / candidate).resolve()
    if must_exist and not result.is_file():
        raise FullCourseEvaluationError(f"{name} does not exist: {result}")
    return result


def load_full_evaluation_config(path: str | Path) -> FullEvaluationConfig:
    """Load the strict YAML-owned all-route evaluation plan."""

    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FullCourseEvaluationError(f"unable to read full-course evaluation config: {error}") from error
    root = _mapping(raw, "full-course evaluation config")
    expected = {
        "schema_version", "curriculum_config", "checkpoint_stage_id", "full_terrain_stage_id",
        "official_stage_ids", "report_path",
    }
    missing = sorted(expected - set(root))
    unknown = sorted(set(root) - expected)
    if missing or unknown:
        raise FullCourseEvaluationError(f"full-course evaluation config keys are invalid: missing={missing}, unknown={unknown}")
    if root["schema_version"] != FULL_EVALUATION_SCHEMA:
        raise FullCourseEvaluationError("unsupported full-course evaluation schema")
    stages = root["official_stage_ids"]
    if not isinstance(stages, list) or not stages:
        raise FullCourseEvaluationError("official_stage_ids must be a non-empty sequence")
    stage_ids = tuple(_string(value, f"official_stage_ids[{index}]") for index, value in enumerate(stages))
    if len(set(stage_ids)) != len(stage_ids):
        raise FullCourseEvaluationError("official_stage_ids must not contain duplicates")
    return FullEvaluationConfig(
        source_path=source,
        curriculum_config_path=_path(source, root["curriculum_config"], "curriculum_config", must_exist=True),
        checkpoint_stage_id=_string(root["checkpoint_stage_id"], "checkpoint_stage_id"),
        full_terrain_stage_id=_string(root["full_terrain_stage_id"], "full_terrain_stage_id"),
        official_stage_ids=stage_ids,
        report_path=_path(source, root["report_path"], "report_path", must_exist=False),
    )


def _validate_route_plan(
    curriculum: CurriculumConfig, plan: FullEvaluationConfig
) -> tuple[StageConfig, tuple[StageConfig, ...]]:
    checkpoint_stage = curriculum.stage(plan.checkpoint_stage_id)
    if not checkpoint_stage.stage_id.startswith("official_"):
        raise FullCourseEvaluationError("checkpoint_stage_id must identify an official route checkpoint")
    stages = tuple(curriculum.stage(stage_id) for stage_id in plan.official_stage_ids)
    if any(not stage.stage_id.startswith("official_") for stage in stages):
        raise FullCourseEvaluationError("official_stage_ids may contain only official route stages")
    adapter_paths = {stage.adapter_config_path for stage in stages}
    if len(adapter_paths) != 1 or None in adapter_paths:
        raise FullCourseEvaluationError("all official full-course routes must use one auditable adapter YAML")
    adapter = load_official_course_config(next(iter(adapter_paths)))
    terrain = load_terrain_curriculum(checkpoint_stage.terrain_curriculum_path)
    full_stage = terrain.stage(plan.full_terrain_stage_id)
    task_to_stage = {adapter.courses[stage.stage_id].task_id: stage.stage_id for stage in stages}
    if len(task_to_stage) != len(stages):
        raise FullCourseEvaluationError("official route stages must map one-to-one to terrain tasks")
    try:
        expected_stage_ids = tuple(task_to_stage[task_id] for task_id in full_stage.task_ids)
    except KeyError as error:
        raise FullCourseEvaluationError("official full terrain stage includes a task without a CUDA route stage") from error
    if plan.official_stage_ids != expected_stage_ids:
        raise FullCourseEvaluationError(
            "official_stage_ids must exactly follow the official_full terrain task order"
        )
    if checkpoint_stage.stage_id not in plan.official_stage_ids:
        raise FullCourseEvaluationError("checkpoint_stage_id must be one of official_stage_ids")
    return checkpoint_stage, stages


def _load_certified_policy(
    checkpoint_path: str | Path, *, curriculum: CurriculumConfig, checkpoint_stage: StageConfig
) -> tuple[Mapping[str, Any], str]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FullCourseEvaluationError(f"checkpoint does not exist: {path}")
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - backend exceptions differ
        raise FullCourseEvaluationError(f"unable to load checkpoint: {error}") from error
    if not isinstance(payload, Mapping):
        raise FullCourseEvaluationError("checkpoint root must be a mapping")
    validate_checkpoint_metadata(payload, config=curriculum, stage=checkpoint_stage)
    return payload, hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_full_course(
    *,
    plan: FullEvaluationConfig,
    checkpoint_path: str | Path,
) -> Mapping[str, Any]:
    """Evaluate one certified policy across every official immutable route."""

    curriculum = load_curriculum_config(plan.curriculum_config_path)
    checkpoint_stage, stages = _validate_route_plan(curriculum, plan)
    checkpoint, checkpoint_sha256 = _load_certified_policy(
        checkpoint_path, curriculum=curriculum, checkpoint_stage=checkpoint_stage
    )
    reports: list[Mapping[str, Any]] = []
    failures: list[Mapping[str, str]] = []
    policy: Any | None = None
    for stage in stages:
        bundle = _external_stage_bundle(curriculum, stage)
        close = _bundle_value(bundle, "close", required=False)
        try:
            batch = _bundle_value(bundle, "batch")
            task = _bundle_value(bundle, "task")
            if policy is None:
                policy = ActorCritic(
                    curriculum.observation_size,
                    curriculum.action_size,
                    hidden_size=curriculum.ppo.hidden_size,
                    initial_action_std=curriculum.ppo.initial_action_std,
                ).to(batch.device)
                policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
                policy.eval()

            def deterministic_action(observation: Any) -> Any:
                actor = getattr(policy, "actor", None)
                if actor is None:
                    raise FullCourseEvaluationError("certified PPO checkpoint does not expose an actor")
                return actor(observation)

            reports.append(
                evaluate_policy_stage(
                    task,
                    deterministic_action,
                    stage,
                    threshold_terrain_stage_id=plan.full_terrain_stage_id,
                )
            )
        except (OfficialCourseAdapterError, RuntimeError, ValueError) as error:
            failures.append({"stage_id": stage.stage_id, "error": str(error)})
        finally:
            if callable(close):
                close()
    if len(reports) != len(stages):
        passed = False
        completion_rate = 0.0
        unsafe_rate = 1.0
        speed_mae = None
        yaw_mae = None
    else:
        completion_rate = min(float(report["completion_rate"]) for report in reports)
        unsafe_rate = max(float(report["unsafe_rate"]) for report in reports)
        speed_mae = max(float(report["speed_mae_mps"]) for report in reports)
        yaw_mae = max(float(report["yaw_mae_rad"]) for report in reports)
        passed = all(report.get("passed") is True for report in reports)
    return {
        "certificate_schema": 1,
        "evaluation_scope": "official_full_all_routes_one_policy",
        "checkpoint_stage_id": checkpoint_stage.stage_id,
        "checkpoint_sha256": checkpoint_sha256,
        "curriculum_config_sha256": curriculum.source_digest,
        "full_terrain_stage_id": plan.full_terrain_stage_id,
        "official_stage_ids": list(plan.official_stage_ids),
        "route_reports": reports,
        "failures": failures,
        "completion_rate": completion_rate,
        "unsafe_rate": unsafe_rate,
        "speed_mae_mps": speed_mae,
        "yaw_mae_rad": yaw_mae,
        "passed": passed,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/official_full_evaluation.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plan = load_full_evaluation_config(args.config)
    report = evaluate_full_course(plan=plan, checkpoint_path=args.checkpoint)
    report_path = plan.report_path if args.report is None else args.report.expanduser().resolve()
    _atomic_json(report_path, report)
    if report["passed"] is not True:
        raise SystemExit(f"official full-course evaluation failed; report={report_path}")
    print(f"official full-course evaluation passed: report={report_path}")


if __name__ == "__main__":
    main()
