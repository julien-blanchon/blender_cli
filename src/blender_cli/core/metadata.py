"""Codec helpers for scene custom-property metadata stored as JSON strings."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

KEY_UID = "_uid"
KEY_TAGS = "_tags"
KEY_ANNOTATIONS = "_annotations"
KEY_PROPS = "_props"
KEY_ASSET_ID = "_asset_id"
KEY_ASSET_PATH = "_asset_path"
KEY_MATERIAL_IDS = "_material_ids"


def _decode_json(raw: object) -> object | None:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def decode_json(raw: object, default: object | None = None) -> object | None:
    parsed = _decode_json(raw)
    return default if parsed is None else parsed


def decode_set(raw: object) -> set[str]:
    parsed = _decode_json(raw)
    if not isinstance(parsed, list):
        return set()
    return {str(v) for v in parsed if str(v)}


def decode_list(raw: object) -> list[str]:
    parsed = _decode_json(raw)
    if not isinstance(parsed, list):
        return []
    return [str(v) for v in parsed if str(v)]


def decode_dict(raw: object) -> dict[str, object]:
    parsed = _decode_json(raw)
    if not isinstance(parsed, dict):
        return {}
    return {str(k): v for k, v in parsed.items()}


def encode_set(values: set[str] | list[str]) -> str:
    return json.dumps(sorted(str(v) for v in values if str(v)))


def encode_list(values: list[str] | tuple[str, ...]) -> str:
    return json.dumps([str(v) for v in values if str(v)])


def encode_dict(values: Mapping[str, object], *, sort_keys: bool = True) -> str:
    return json.dumps(dict(values), sort_keys=sort_keys)


def encode_json(values: object, *, sort_keys: bool = True) -> str:
    return json.dumps(values, sort_keys=sort_keys)
