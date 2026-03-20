"""Session management — undo/redo with history tracking for ProjectFile."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from blender_cli.project.project_file import ProjectFile


class Session:
    """
    Undo/redo session for a ProjectFile.

    Maintains a LIFO undo stack (max 50 snapshots) and a redo stack.
    The redo stack is cleared whenever a new snapshot is taken.
    """

    MAX_UNDO = 50

    def __init__(self, project: ProjectFile) -> None:
        from blender_cli.project.project_file import ProjectFile

        if not isinstance(project, ProjectFile):
            msg = "Session requires a ProjectFile instance"
            raise TypeError(msg)
        self._project = project
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []
        self._modified = False

    # -- Properties ----------------------------------------------------------

    @property
    def modified(self) -> bool:
        return self._modified

    @property
    def project(self) -> ProjectFile:
        return self._project

    # -- Snapshot / Undo / Redo ----------------------------------------------

    def snapshot(self, description: str = "unnamed") -> None:
        """Save a deep-copy snapshot of the current project state."""
        entry = {
            "data": copy.deepcopy(self._project.data),
            "description": description,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._undo_stack.append(entry)
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        # New mutation clears redo stack
        self._redo_stack.clear()
        self._modified = True

    def undo(self) -> str:
        """Revert to the last snapshot. Returns description of undone action."""
        if not self._undo_stack:
            msg = "Nothing to undo"
            raise IndexError(msg)
        # Save current state to redo stack
        redo_entry = {
            "data": copy.deepcopy(self._project.data),
            "description": self._undo_stack[-1]["description"],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._redo_stack.append(redo_entry)
        # Restore from undo stack
        entry = self._undo_stack.pop()
        self._project.data = entry["data"]
        self._modified = True
        return entry["description"]

    def redo(self) -> str:
        """Re-apply the last undone action. Returns description."""
        if not self._redo_stack:
            msg = "Nothing to redo"
            raise IndexError(msg)
        # Save current state to undo stack before redo
        undo_entry = {
            "data": copy.deepcopy(self._project.data),
            "description": self._redo_stack[-1]["description"],
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._undo_stack.append(undo_entry)
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        # Restore from redo stack
        entry = self._redo_stack.pop()
        self._project.data = entry["data"]
        self._modified = True
        return entry["description"]

    def clear_modified(self) -> None:
        """Clear the modified flag (called after save)."""
        self._modified = False

    # -- Info ----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return session status dict."""
        return {
            "has_project": True,
            "project_path": str(self._project.path) if self._project.path else None,
            "modified": self._modified,
            "undo_count": len(self._undo_stack),
            "redo_count": len(self._redo_stack),
            "scene_name": self._project.name,
        }

    def list_history(self) -> list[dict[str, Any]]:
        """Return list of undo entries (oldest first) with timestamps and descriptions."""
        return [
            {
                "index": i,
                "description": entry["description"],
                "timestamp": entry["timestamp"],
            }
            for i, entry in enumerate(self._undo_stack)
        ]
