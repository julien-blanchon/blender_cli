"""Geometry primitives — fields, heightfields, masks, splines, point sets."""

from blender_cli.geometry.field2d import CombineOp, Field2D, GradientDirection
from blender_cli.geometry.heightfield import (
    ErosionType,
    FalloffCurve,
    Heightfield,
    NoiseType,
    StampOp,
    StampShape,
)
from blender_cli.geometry.mask import HeightfieldMode, Mask
from blender_cli.geometry.pointset import WILDCARD, PointAttrs, PointSet
from blender_cli.geometry.spline import Spline
from blender_cli.geometry.spline_ops import SplineOp

__all__ = [
    "WILDCARD",
    "CombineOp",
    "ErosionType",
    "FalloffCurve",
    "Field2D",
    "GradientDirection",
    "Heightfield",
    "HeightfieldMode",
    "Mask",
    "NoiseType",
    "PointAttrs",
    "PointSet",
    "Spline",
    "SplineOp",
    "StampOp",
    "StampShape",
]
