"""Scene subpackage — core scene graph, entities, anchors, selection, primitives, instances."""

from blender_cli.scene.anchor import Anchor
from blender_cli.scene.entity import Entity, SnapSpec, as_entity, unwrap_entity
from blender_cli.scene.instances import Instances
from blender_cli.scene.primitives import box, cone, cylinder, plane, sphere, torus
from blender_cli.scene.scene import Scene
from blender_cli.scene.selection import Selection, Transform, parse_query

__all__ = [
    "Anchor",
    "Entity",
    "Instances",
    "Scene",
    "Selection",
    "SnapSpec",
    "Transform",
    "as_entity",
    "box",
    "cone",
    "cylinder",
    "parse_query",
    "plane",
    "sphere",
    "torus",
    "unwrap_entity",
]
