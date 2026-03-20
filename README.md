# blender-cli

Agent-friendly procedural map generation SDK built on Blender's Python API.

`blender-cli` provides a **Python SDK** and a **CLI** for building 3D scenes declaratively through a `project.json` file — procedural terrain, PBR materials, object placement with snap/raycast, GPU-instanced scatter, cameras, lights, and Bevy/Blenvy ECS component embedding — then exports to GLB.

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install from source
uv pip install .

# Or run directly with uvx (no install needed)
uvx --from . blender-cli

# From a git repo
uvx --from git+https://github.com/julien-blanchon/blender_cli blender-cli
```

## Quick start

### CLI

```bash
# Create a project
blender-cli project new --name "Alpine" --profile hd1080p -o project.json

# Add materials
blender-cli material add --project project.json grass --pbr-folder ./textures/grass/ --tiling 4 4

# Build terrain
blender-cli terrain init --project project.json --width 512 --height 512 --seed 42
blender-cli terrain op --project project.json noise type=fbm amp=15 freq=0.008 octaves=6
blender-cli terrain op --project project.json smooth radius=3 iters=1
blender-cli terrain material --project project.json grass

# Place objects (snapped to terrain)
blender-cli project add-object --project project.json Wall \
  --primitive box --primitive-params '{"size":[10,0.3,3]}' \
  --at 50 50 0 --material rock --snap -Z --tags building

# Add camera and light
blender-cli project add-camera --project project.json Main --at 0 -50 30 --lens 35
blender-cli project add-light --project project.json Sun --type SUN --energy 5.0

# Export to GLB
blender-cli project export --project project.json --out scene.glb

# Render a preview
blender-cli render still --glb scene.glb --project project.json --preset iso --out preview.png
```

### Python SDK

```python
from blender_cli import ProjectFile

pf = ProjectFile.new("Alpine", profile="hd1080p")

# Materials
pf.add_material("grass", pbr_folder="./textures/grass/", tiling=[4, 4])
pf.add_material("rock", pbr_folder="./textures/rock/", tiling=[2, 2])

# Terrain
pf.set_terrain(512, 512, seed=42)
pf.terrain_op("noise", type="fbm", amp=12.0, freq=0.008, octaves=6)
pf.terrain_op("smooth", radius=3, iters=1)
pf.terrain_op("erode", type="hydraulic", iterations=40)
pf.set_terrain_material("grass")

# Anchors (named spatial references)
pf.add_anchor("village", [256, 256, 0], annotation="center")

# Objects (snapped to terrain at export)
pf.add_object("Cottage", asset_path="./assets/cottage.glb",
              location=pf.anchor_pos("village"), tags=["building"], snap=True)

# Camera, light
pf.add_camera("Main", location=[0, -50, 30], look_at=[256, 256, 0], lens=35)
pf.add_light("Sun", "SUN", energy=5.0, rotation=[0.5, 0.0, 0.0])

# Export
pf.save("project.json")
pf.export_glb("scene.glb")
```

## Architecture

```
project.json  ──[export_glb]──►  scene.glb
     ▲                              │
     │                              │
     └────────[import_glb]──────────┘
```

Everything goes through **`project.json`** as the single source of truth. Edit it via the Python SDK (`ProjectFile`) or the CLI (`blender-cli`), then export to GLB. You can also import an existing GLB back into a project file.

## Features

### Terrain

Procedural heightfield terrain built from a chain of operations:

| Operation        | Effect                    |
| ---------------- | ------------------------- |
| `flat`           | Start flat at elevation z |
| `noise`          | Add fbm/ridged noise      |
| `smooth`         | Gaussian blur             |
| `terrace`        | Plateau effect            |
| `clamp`          | Clamp height range        |
| `stamp`          | Local circular/ring edit  |
| `erode`          | Hydraulic/thermal erosion |
| `radial_falloff` | Island/crater falloff     |
| `remap_curve`    | Height remapping curve    |

### Materials

PBR texture folders (auto-detects diffuse, normal, roughness, AO, displacement) or flat colors with metallic/roughness control.

### Objects & Primitives

Primitives: `box`, `plane`, `cylinder`, `sphere`, `cone`, `torus`. Also supports prefab GLB assets and empties.

**Snap system** — raycast objects onto geometry at export time:

| Axis | Use case      | Policy    | Effect                   |
| ---- | ------------- | --------- | ------------------------ |
| `-Z` | Floor/terrain | `FIRST`   | Sit on first surface hit |
| `+Z` | Ceiling mount | `ORIENT`  | Align to surface normal  |
| `+X` | Wall mount    | `HIGHEST` | Highest hit Z            |

Filter snapping with `target_tags` and `exclude_tags`.

### Anchors

Named spatial reference points. Place objects, cameras, and lights relative to anchors instead of hardcoded coordinates:

```python
pf.add_anchor("gate", [100, 100, 0])
pf.add_object("Tower", location=pf.anchor_pos("gate", [5, 0, 0]), snap=True)
```

### Instances

GPU-instanced scatter groups for vegetation, rocks, and props with per-point rotation and scale.

### Cameras

Static cameras or animated paths (orbit, dolly, flyover). Ghost cameras are excluded from GLB export and used only for debug renders.

### Lights

SUN, POINT, SPOT, AREA light types. Ghost lights for debug previews.

### Bevy / Blenvy Integration

Embed Bevy ECS components directly in the GLB. The SDK validates component names against a Bevy `registry.json`:

```bash
blender-cli blenvy set-registry --project project.json assets/registry.json
blender-cli blenvy add-component --project project.json Ground RigidBody Static
blender-cli blenvy validate --registry assets/registry.json --project project.json
```

### Render & Debug

Render stills from named cameras or presets (`top`, `iso`). Supports:

- Tag-based show/hide (`--hide-tag ceiling` for dollhouse views)
- Wireframe mode
- Highlight + ghost mode
- Decomposition render (flat colors per object/tag)
- Multi-pass (beauty, depth, normal)

### Session

Undo/redo with named snapshots.

### REPL

Interactive prompt for exploring scenes:

```bash
blender-cli repl --project project.json
```

## Render profiles

| Profile             | Engine | Resolution | Samples | Use case            |
| ------------------- | ------ | ---------- | ------- | ------------------- |
| `default`           | EEVEE  | 1920x1080  | 64      | General development |
| `preview`           | EEVEE  | 960x540    | 16      | Quick iteration     |
| `hd1080p`           | CYCLES | 1920x1080  | 128     | High quality        |
| `4k`                | CYCLES | 3840x2160  | 256     | Ultra quality       |
| `product_render`    | CYCLES | 2048x2048  | 512     | Transparent bg      |
| `animation_preview` | EEVEE  | 1280x720   | 16      | Animation testing   |

## CLI reference

```
blender-cli [--project FILE] [--json] COMMAND
```

| Command      | Description                               |
| ------------ | ----------------------------------------- |
| `project`    | Create, info, export, import projects     |
| `terrain`    | Init, add ops, set mesh/material          |
| `material`   | Add/list PBR or flat-color materials      |
| `anchor`     | Add/list/remove spatial anchors           |
| `object`     | Add/remove/manage objects                 |
| `instance`   | Add/list/remove GPU-instanced groups      |
| `camera`     | Add cameras and animated paths            |
| `render`     | Render stills, focus shots, decomposition |
| `world`      | Set background color or HDRI              |
| `blenvy`     | Bevy component validation and embedding   |
| `session`    | Undo/redo/history                         |
| `select`     | Query objects with DSL                    |
| `inspect`    | Inspect GLB scene contents                |
| `stats`      | Scene statistics                          |
| `repl`       | Interactive REPL                          |
| `align`      | Image-to-3D alignment pipeline            |
| `animation`  | Keyframe animation                        |
| `modifier`   | Geometry modifiers                        |
| `op`         | Low-level Blender operations              |
| `run`        | Execute build scripts                     |
| `measure`    | Measure distances and dimensions          |
| `raycast`    | Raycast queries                           |
| `candidates` | Placement candidate generation            |
| `assets`     | Asset management                          |
| `manifest`   | Scene manifest operations                 |

Pass `--json` for machine-readable JSON output on all commands.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Type check
uv run pyright
```

## License

MIT
