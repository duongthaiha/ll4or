"""Solver dependency bootstrap — check for and (optionally) install common
Python packages used by LLM-generated OR solver code.

The list of packages lives in ``requirements-solver.txt`` at the repo root.
Each line maps a pip package name (optionally == version) to an import-name
using simple heuristics. A small hand-maintained alias table covers the cases
where pip name ≠ import name (e.g. ``scikit-learn`` → ``sklearn``).
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
from pathlib import Path

from src.execution.sandbox import _PYTHON

log = logging.getLogger(__name__)

# pip-name → import-name overrides (only needed when they differ)
_IMPORT_ALIASES: dict[str, str] = {
    "scikit-learn": "sklearn",
    "python-mip": "mip",
    "opencv-python": "cv2",
}


def _repo_root() -> Path:
    # src/execution/bootstrap.py → repo root is two parents up
    return Path(__file__).resolve().parent.parent.parent


def _requirements_file() -> Path:
    return _repo_root() / "requirements-solver.txt"


def _parse_requirements(path: Path) -> list[str]:
    """Return list of pip package specs, stripped of comments/blank lines."""
    if not path.exists():
        return []
    specs: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            specs.append(line)
    return specs


def _pip_to_import(spec: str) -> str:
    """Map a pip spec (e.g. 'scikit-learn>=1.0') to its import name."""
    name = spec.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("<")[0].split(">")[0]
    name = name.strip().lower()
    if name in _IMPORT_ALIASES:
        return _IMPORT_ALIASES[name]
    # Standard normalization: hyphens and dots → underscores
    return name.replace("-", "_").replace(".", "_")


def _is_installed(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def check_solver_deps() -> tuple[list[str], list[str]]:
    """Return (installed, missing) pip-spec lists from requirements-solver.txt."""
    specs = _parse_requirements(_requirements_file())
    installed: list[str] = []
    missing: list[str] = []
    for spec in specs:
        if _is_installed(_pip_to_import(spec)):
            installed.append(spec)
        else:
            missing.append(spec)
    return installed, missing


def log_missing_solver_deps() -> list[str]:
    """Emit a warning listing missing solver packages. Returns the missing list."""
    installed, missing = check_solver_deps()
    if not _requirements_file().exists():
        log.debug("No requirements-solver.txt found; skipping dep check")
        return []
    if missing:
        log.warning(
            "Solver packages missing (%d of %d): %s — "
            "run `pip install -r requirements-solver.txt` or pass "
            "`--install-solver-deps` to install now.",
            len(missing), len(installed) + len(missing), ", ".join(missing),
        )
    else:
        log.info(
            "All %d solver packages from requirements-solver.txt are installed.",
            len(installed),
        )
    return missing


def install_solver_deps(missing_only: bool = True) -> bool:
    """Install solver packages via pip. Returns True on success.

    When ``missing_only`` is True, only installs packages whose import fails.
    """
    req_file = _requirements_file()
    if not req_file.exists():
        log.warning("No requirements-solver.txt found at %s", req_file)
        return False

    if missing_only:
        _, specs = check_solver_deps()
        if not specs:
            log.info("All solver packages already installed — nothing to do.")
            return True
        log.info("Installing %d missing solver packages: %s", len(specs), ", ".join(specs))
        cmd = [_PYTHON, "-m", "pip", "install", *specs]
    else:
        log.info("Installing all solver packages from %s", req_file)
        cmd = [_PYTHON, "-m", "pip", "install", "-r", str(req_file)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log.error("pip install timed out after 600s")
        return False
    except FileNotFoundError:
        log.error("Could not locate python interpreter for pip install: %s", _PYTHON)
        return False

    if result.returncode != 0:
        log.error(
            "pip install failed (exit %d):\nstdout: %s\nstderr: %s",
            result.returncode, result.stdout[-2000:], result.stderr[-2000:],
        )
        return False

    log.info("Solver packages installed successfully.")
    _, still_missing = check_solver_deps()
    if still_missing:
        log.warning(
            "After install, still missing: %s (pip reports success but import fails)",
            ", ".join(still_missing),
        )
    return True
