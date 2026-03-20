"""Shared diagnostics/logging primitives for SDK internals."""

from __future__ import annotations

import logging

# Library modules should log through this namespace and avoid direct prints.
logger = logging.getLogger("blender_cli")
logger.addHandler(logging.NullHandler())
