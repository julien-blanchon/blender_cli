"""Shared fal.ai client helpers for alignment workflows."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging

_INSTALL_HINT = "Install it with `uv add fal-client` (or `pip install fal-client`)."


def require_fal_client(feature: str = "Alignment fal.ai integration") -> Any:
    """Import ``fal_client`` with a consistent error message."""
    try:
        return importlib.import_module("fal_client")
    except ImportError as exc:  # pragma: no cover - dependency issue
        msg = f"{feature} requires fal-client. {_INSTALL_HINT}"
        raise RuntimeError(msg) from exc


def upload_file(path: str | Path) -> str:
    """Upload a local file to Fal storage and return the remote URL."""
    client = require_fal_client("Alignment generation")
    return str(client.upload_file(str(Path(path))))


def upload_image(image: Any, *, fmt: str = "png") -> str:
    """Upload a PIL image to Fal storage and return the remote URL."""
    client = require_fal_client("Alignment fal.ai image upload")
    return str(client.upload_image(image, fmt))


def subscribe(
    application: str,
    *,
    arguments: dict[str, Any],
    with_logs: bool = False,
    logger: logging.Logger | None = None,
    log_prefix: str = "",
    client_timeout: float | None = None,
) -> dict[str, Any]:
    """Call ``fal_client.subscribe`` with optional queue log forwarding."""
    client = require_fal_client("Alignment fal.ai inference")
    subscribe_kwargs: dict[str, Any] = {
        "arguments": arguments,
        "with_logs": with_logs,
    }

    if with_logs and logger is not None:
        in_progress_type = getattr(client, "InProgress", None)

        def _on_queue_update(update: object) -> None:
            if in_progress_type is not None and not isinstance(
                update, in_progress_type
            ):
                return
            entries = getattr(update, "logs", None) or []
            for entry in entries:
                if isinstance(entry, dict):
                    message = entry.get("message")
                else:
                    message = getattr(entry, "message", None)
                if message:
                    logger.info("%s%s", log_prefix, message)

        subscribe_kwargs["on_queue_update"] = _on_queue_update

    if client_timeout is not None:
        subscribe_kwargs["client_timeout"] = client_timeout

    return client.subscribe(application, **subscribe_kwargs)
