"""
Runner — execute example generation scripts via BuildContext.

Usage:
    uv run python -m blender_cli.runner examples/alpine_valley/generation_001.py
"""

from __future__ import annotations


import importlib.util
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from blender_cli.build.context import BuildContext

if TYPE_CHECKING:
    from types import ModuleType


def _load_script(script_path: Path) -> ModuleType:
    """Import a Python file as a module."""
    spec = importlib.util.spec_from_file_location("_gen_script", script_path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load script: {script_path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _step_index(script_path: Path) -> int:
    """Extract step index from filename like ``generation_001.py`` -> 1."""
    m = re.search(r"(\d+)", script_path.stem)
    return int(m.group(1)) if m else 1


def run_script(script_path: str | Path, seed: int = 0) -> None:
    """Run a generation script's ``run(ctx)`` function."""
    script = Path(script_path).resolve()
    if not script.is_file():
        msg = f"Script not found: {script}"
        raise FileNotFoundError(msg)

    project_dir = script.parent
    step = _step_index(script)
    ctx = BuildContext(project_dir, step, seed=seed)

    mod = _load_script(script)
    if not hasattr(mod, "run"):
        msg = f"Script {script.name} has no 'run(ctx)' function"
        raise AttributeError(msg)
    mod.run(ctx)


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m blender_cli.runner <script.py> [--seed N]")
        sys.exit(1)

    script = sys.argv[1]
    seed = 0
    if "--seed" in sys.argv:
        idx = sys.argv.index("--seed")
        seed = int(sys.argv[idx + 1])

    run_script(script, seed=seed)


if __name__ == "__main__":
    main()
