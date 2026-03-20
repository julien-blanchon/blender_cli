"""Interactive REPL for blender_cli."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


class ReplSession:
    """
    Programmatic REPL session for blender_cli.

    Wraps the Click CLI group so every blender_cli command can be invoked
    interactively without the ``blender_cli`` prefix.
    """

    HISTORY_DIR = Path.home() / ".blender_cli"
    HISTORY_FILE = HISTORY_DIR / "history"

    def __init__(self, cli_group: Any, project_path: str | None = None) -> None:
        self._cli = cli_group
        self._project_path = project_path

    @property
    def prompt_text(self) -> str:
        if self._project_path:
            name = Path(self._project_path).stem
            return f"blender_cli [{name}] > "
        return "blender_cli > "

    # ------------------------------------------------------------------
    # Low-level: run a single command line (no prompt_toolkit needed)
    # ------------------------------------------------------------------

    def run_one(self, line: str) -> bool:
        """Execute a single command line. Returns False to signal exit."""
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        if line in {"quit", "exit", "q"}:
            return False
        if line == "help":
            args = ["--help"]
        else:
            try:
                args = shlex.split(line)
            except ValueError as exc:
                import click

                click.echo(f"Parse error: {exc}", err=True)
                return True

        # Inject --project if set and not already present
        if self._project_path and "--project" not in args:
            args = ["--project", self._project_path, *args]

        import click

        try:
            self._cli(args, standalone_mode=False)
        except SystemExit:
            pass
        except click.ClickException as exc:
            click.echo(f"Error: {exc.format_message()}", err=True)
        except Exception as exc:
            click.echo(f"Error ({type(exc).__name__}): {exc}", err=True)
        return True

    # ------------------------------------------------------------------
    # Interactive loop (with prompt_toolkit when available)
    # ------------------------------------------------------------------

    def loop(self) -> None:
        """Run the interactive REPL loop."""
        self.HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        try:
            self._loop_prompt_toolkit()
        except ImportError:
            self._loop_fallback()

    def _get_command_names(self) -> list[str]:
        """Collect all top-level command names for tab completion."""
        import click

        if isinstance(self._cli, click.Group):
            ctx = click.Context(self._cli)
            return sorted(self._cli.list_commands(ctx))
        return []

    def _loop_prompt_toolkit(self) -> None:
        """REPL loop using prompt_toolkit for history, tab-completion, etc."""
        from prompt_toolkit import PromptSession  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.completion import WordCompleter  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.history import FileHistory  # pyright: ignore[reportMissingImports]

        commands = [*self._get_command_names(), "help", "quit", "exit", "q"]
        completer = WordCompleter(commands, ignore_case=True)
        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(self.HISTORY_FILE)),
            completer=completer,
        )

        while True:
            try:
                line = session.prompt(self.prompt_text)
            except (EOFError, KeyboardInterrupt):
                break
            if not self.run_one(line):
                break

    def _loop_fallback(self) -> None:
        """Simple input() fallback when prompt_toolkit is unavailable."""
        while True:
            try:
                line = input(self.prompt_text)
            except (EOFError, KeyboardInterrupt):
                break
            if not self.run_one(line):
                break
