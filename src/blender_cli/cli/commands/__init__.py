"""
Command registration for blender_cli subcommands.

Each command module defines module-level Click groups/commands.
``register_all`` adds them to the main CLI group via ``add_command()``.
"""

import click

from blender_cli.cli.commands.align import align
from blender_cli.cli.commands.anchor_cmd import anchor_cmd
from blender_cli.cli.commands.blenvy_cmd import blenvy_cmd
from blender_cli.cli.commands.animation_cmd import animation
from blender_cli.cli.commands.assets import assets
from blender_cli.cli.commands.camera_cmd import camera_cmd
from blender_cli.cli.commands.candidates import candidates
from blender_cli.cli.commands.inspect import inspect
from blender_cli.cli.commands.instance_cmd import instance_cmd
from blender_cli.cli.commands.manifest import manifest
from blender_cli.cli.commands.material_cmd import material_cmd
from blender_cli.cli.commands.measure import measure
from blender_cli.cli.commands.modifier_cmd import modifier
from blender_cli.cli.commands.object_cmd import object_cmd
from blender_cli.cli.commands.op import op
from blender_cli.cli.commands.project import project
from blender_cli.cli.commands.raycast import raycast
from blender_cli.cli.commands.render import render
from blender_cli.cli.commands.repl_cmd import repl_cmd
from blender_cli.cli.commands.run import run
from blender_cli.cli.commands.select import select
from blender_cli.cli.commands.session_cmd import session
from blender_cli.cli.commands.stats import stats
from blender_cli.cli.commands.terrain_cmd import terrain_cmd
from blender_cli.cli.commands.world_cmd import world_cmd

__all__ = ["register_all"]


def register_all(main: click.Group) -> None:
    """Register all subcommands with the main CLI group."""
    main.add_command(align)
    main.add_command(anchor_cmd)
    main.add_command(blenvy_cmd)
    main.add_command(animation)
    main.add_command(camera_cmd)
    main.add_command(render)
    main.add_command(candidates)
    main.add_command(assets)
    main.add_command(inspect)
    main.add_command(instance_cmd)
    main.add_command(select)
    main.add_command(material_cmd)
    main.add_command(measure)
    main.add_command(stats)
    main.add_command(manifest)
    main.add_command(raycast)
    main.add_command(modifier)
    main.add_command(object_cmd)
    main.add_command(op)
    main.add_command(project)
    main.add_command(repl_cmd)
    main.add_command(run)
    main.add_command(session)
    main.add_command(terrain_cmd)
    main.add_command(world_cmd)
