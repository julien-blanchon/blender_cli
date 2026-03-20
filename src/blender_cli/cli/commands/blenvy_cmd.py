"""Blenvy registry CLI — list, inspect, validate, and manage Bevy components."""

from __future__ import annotations

import json

import click

from blender_cli.blenvy_registry import (
    AmbiguousComponentError,
    BevyRegistry,
    UnknownComponentError,
)
from blender_cli.project.project_file import ProjectFile


@click.group("blenvy")
def blenvy_cmd() -> None:
    """Blenvy Bevy component registry — list, inspect, validate."""


def _load_registry(registry: str | None, project_path: str | None) -> BevyRegistry:
    """Load registry from explicit path, project config, or raise."""
    if registry:
        return BevyRegistry.load(registry)
    if project_path:
        pf = ProjectFile.load(project_path)
        rpath = pf.data.get("blenvy", {}).get("registry_path")
        if rpath:
            return BevyRegistry.load(rpath)
    msg = "No registry provided. Use --registry or set registry path in project with 'blenvy set-registry'."
    raise click.UsageError(msg)


@blenvy_cmd.command("list-components")
@click.option("--registry", type=click.Path(exists=True), default=None)
@click.option("--project", "project_path", type=click.Path(exists=True), default=None)
@click.option("--prefix", default=None, help="Filter by Rust module prefix (e.g. avian3d).")
@click.option("--search", default=None, help="Filter short names containing this substring.")
def list_components(
    registry: str | None,
    project_path: str | None,
    prefix: str | None,
    search: str | None,
) -> None:
    """List all available Bevy ECS components from the registry."""
    reg = _load_registry(registry, project_path)
    comps = reg.components(prefix=prefix)
    if search:
        needle = search.lower()
        comps = [c for c in comps if needle in c.short_name.lower() or needle in c.long_name.lower()]
    rows = [
        {
            "short_name": c.short_name,
            "long_name": c.long_name,
            "type_info": c.type_info,
        }
        for c in comps
    ]
    click.echo(json.dumps({"status": "ok", "count": len(rows), "components": rows}, indent=2))


@blenvy_cmd.command("inspect")
@click.option("--registry", type=click.Path(exists=True), default=None)
@click.option("--project", "project_path", type=click.Path(exists=True), default=None)
@click.argument("name")
def inspect_component(registry: str | None, project_path: str | None, name: str) -> None:
    """Show full schema for a Bevy component. Accepts short or full name."""
    reg = _load_registry(registry, project_path)
    try:
        full_path = reg.resolve(name)
    except (UnknownComponentError, AmbiguousComponentError) as exc:
        suggestions = getattr(exc, "suggestions", None) or getattr(exc, "candidates", None) or []
        click.echo(
            json.dumps({"status": "error", "error": str(exc), "suggestions": suggestions}, indent=2),
            err=True,
        )
        raise SystemExit(1) from None
    info = reg.find(full_path)
    assert info is not None
    result: dict = {
        "status": "ok",
        "short_name": info.short_name,
        "long_name": info.long_name,
        "type_info": info.type_info,
    }
    if info.variants:
        result["variants"] = info.variants
    if info.fields:
        result["fields"] = info.fields
    if info.required_fields:
        result["required"] = info.required_fields
    result["schema"] = info.schema
    click.echo(json.dumps(result, indent=2))


@blenvy_cmd.command("validate")
@click.option("--registry", type=click.Path(exists=True), default=None)
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def validate_project(registry: str | None, project_path: str) -> None:
    """Validate all bevy_components in a project against the registry."""
    reg = _load_registry(registry, project_path)
    pf = ProjectFile.load(project_path)
    issues: list[dict] = []
    # Check objects, cameras, and lights for bevy_components
    items = [
        *[(obj, "object") for obj in pf.data.get("objects", [])],
        *[(cam, "camera") for cam in pf.data.get("cameras", [])],
        *[(lgt, "light") for lgt in pf.data.get("lights", [])],
    ]
    for item, kind in items:
        bcomps = item.get("bevy_components")
        if not bcomps:
            continue
        item_name = item.get("name", item.get("uid", "?"))
        for comp_name, comp_value in bcomps.items():
            info = reg.find(comp_name)
            if info is None:
                suggestions = reg.suggest(comp_name)
                issues.append({
                    kind: item_name,
                    "component": comp_name,
                    "issue": "unknown_component",
                    "suggestions": suggestions,
                })
                continue
            warnings = reg.validate_value(comp_name, comp_value)
            for w in warnings:
                issues.append({
                    kind: item_name,
                    "component": comp_name,
                    "issue": "value_warning",
                    "detail": w,
                })
    status = "ok" if not issues else "warnings"
    click.echo(json.dumps({"status": status, "issues_count": len(issues), "issues": issues}, indent=2))


@blenvy_cmd.command("add-component")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.option("--registry", type=click.Path(exists=True), default=None)
@click.argument("object_name")
@click.argument("component")
@click.argument("value", default="")
def add_component(
    project_path: str,
    registry: str | None,
    object_name: str,
    component: str,
    value: str,
) -> None:
    """Add a Bevy component to an object. Resolves short names via the registry.

    VALUE is a RON string or JSON. Empty for unit components.
    """
    pf = ProjectFile.load(project_path)

    # Resolve component name if registry available
    resolved_name = component
    if registry or pf.data.get("blenvy", {}).get("registry_path"):
        try:
            reg = _load_registry(registry, project_path)
            resolved_name = reg.resolve(component)
        except (UnknownComponentError, AmbiguousComponentError) as exc:
            suggestions = getattr(exc, "suggestions", None) or getattr(exc, "candidates", None) or []
            click.echo(
                json.dumps({"status": "error", "error": str(exc), "suggestions": suggestions}, indent=2),
                err=True,
            )
            raise SystemExit(1) from None

    # Find the object
    target = None
    for obj in pf.data.get("objects", []):
        if obj.get("name") == object_name:
            target = obj
            break
    if target is None:
        click.echo(
            json.dumps({"status": "error", "error": f"Object {object_name!r} not found in project"}, indent=2),
            err=True,
        )
        raise SystemExit(1)

    # Parse value: try JSON first, then treat as raw string
    parsed_value: str | dict | list | None
    if not value:
        parsed_value = None
    else:
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value

    # Write component
    if "bevy_components" not in target:
        target["bevy_components"] = {}

    # Convert value to RON string for storage
    from blender_cli.blenvy import to_ron

    ron_str = to_ron(parsed_value)
    target["bevy_components"][resolved_name] = ron_str
    pf.save()
    click.echo(json.dumps({
        "status": "ok",
        "action": "add_component",
        "object": object_name,
        "component": resolved_name,
        "value_ron": ron_str,
    }, indent=2))


@blenvy_cmd.command("set-registry")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("registry_path")
def set_registry(project_path: str, registry_path: str) -> None:
    """Store the registry path in the project for automatic loading."""
    pf = ProjectFile.load(project_path)
    if "blenvy" not in pf.data:
        pf.data["blenvy"] = {}
    pf.data["blenvy"]["registry_path"] = registry_path
    pf.save()
    click.echo(json.dumps({
        "status": "ok",
        "action": "set_registry",
        "registry_path": registry_path,
    }, indent=2))
