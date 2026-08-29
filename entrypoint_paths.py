"""Path helpers for command-line entry points.

Project commands are often launched with an absolute script path while the
shell remains in the parent workspace.  Configuration paths therefore need a
deterministic project-directory fallback instead of relying on the process
working directory.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def project_path(*parts: str) -> Path:
    """Return an absolute path rooted at the GPU sandbox directory."""

    return PROJECT_DIR.joinpath(*parts)


def resolve_cli_input(value: str | Path) -> Path:
    """Resolve an input path from CWD, then from the script directory.

    Existing paths in the caller's working directory retain precedence.  This
    preserves explicit workspace-relative paths while making common
    ``configs/...`` arguments work when a script is launched from its parent
    directory.  For a missing path we keep the CWD interpretation so the
    eventual error names the path the caller actually supplied.
    """

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    project_candidate = (PROJECT_DIR / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    # Preserve the caller's original relative spelling for a missing input.
    # Downstream loaders resolve it against the actual CWD and retain their
    # established error messages/API behavior.
    return candidate


__all__ = ["PROJECT_DIR", "project_path", "resolve_cli_input"]
