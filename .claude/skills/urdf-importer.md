# URDF Importer — Isaac Lab

Import URDF robot descriptions into Isaac Lab as USD assets. The importer
converts URDF to USD, applies physics schemas, joint drives, and collision
properties, then optionally restructures the output via the asset transformer.

**Requires Isaac Sim** — the URDF converter depends on the
`isaacsim.asset.importer.urdf` extension and USD/PhysX libraries. It does
**not** work in Newton-only (Kit-less) mode.

## Architecture

```
URDF file
  │
  ├─ (optional) merge_fixed_joints: XML pre-processing removes fixed joints,
  │   merges child visuals/collisions/inertials into parent links
  │
  ▼
urdf-usd-converter (Converter)  →  intermediate .usd
  │
  ├─ remove_custom_scopes
  ├─ add_rigid_body_schemas
  ├─ add_joint_schemas
  ├─ collision_from_visuals (if enabled)
  ├─ enable_self_collision
  ├─ fix_base (add FixedJoint to root link)
  ├─ link_density (set default density)
  └─ joint_drives (type, target, gains)
  │
  ▼
asset transformer (isaacsim.asset.transformer.rules)  →  final structured .usda
  │
  └─ fix ArticulationRootAPI placement (if fix_base)
```

## Two ways to import URDF

### 1. Direct converter (standalone scripts, tests)

```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

cfg = UrdfConverterCfg(
    asset_path="/path/to/robot.urdf",
    fix_base=True,
    merge_fixed_joints=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=100.0,
            damping=10.0,
        ),
    ),
)
converter = UrdfConverter(cfg)
usd_path = converter.usd_path  # path to generated .usda
```

### 2. UrdfFileCfg (in asset/environment configs — preferred)

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

robot = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/path/to/robot.urdf",
        fix_base=True,
        merge_fixed_joints=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None, damping=None,
            ),
        ),
    ),
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,
            damping=10.0,
        ),
    },
)
```

## Configuration reference

### `UrdfConverterCfg` (extends `AssetConverterBaseCfg`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `asset_path` | `str` | MISSING | Absolute path to the `.urdf` file |
| `usd_dir` | `str \| None` | `None` | Output directory. `None` → `/tmp/IsaacLab/usd_{date}_{time}_{random}` |
| `force_usd_conversion` | `bool` | `False` | Force re-conversion even if USD exists and config hasn't changed |
| `fix_base` | `bool` | MISSING | Add a FixedJoint from world to root link (required field) |
| `merge_fixed_joints` | `bool` | `True` | XML pre-process: merge fixed-joint child links into parents |
| `link_density` | `float` | `0.0` | Default density (kg/m^3) for links missing inertial properties |
| `collision_from_visuals` | `bool` | `False` | Generate collision geometry from visual meshes |
| `collision_type` | `str` | `"Convex Hull"` | `"Convex Hull"` or `"Convex Decomposition"` |
| `self_collision` | `bool` | `False` | Enable self-collision between links |
| `joint_drive` | `JointDriveCfg \| None` | `JointDriveCfg()` | Joint drive settings. Set `None` for URDFs without joints |
| `make_instanceable` | `bool` | `True` | Make USD instanceable (saves memory with multiple copies) |

### `JointDriveCfg`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drive_type` | `str \| dict[str,str]` | `"force"` | `"force"` or `"acceleration"`. Dict maps regex→type |
| `target_type` | `str \| dict[str,str]` | `"position"` | `"none"`, `"position"`, or `"velocity"`. Dict maps regex→type |
| `gains` | `PDGainsCfg` | `PDGainsCfg()` | PD gain configuration |

### `PDGainsCfg`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stiffness` | `float \| dict[str,float] \| None` | `None` | Stiffness in Nm/rad (revolute) or N/m (prismatic). `None` preserves importer values. Dict maps joint regex→value |
| `damping` | `float \| dict[str,float] \| None` | `None` | Damping in Nm/(rad/s) or N/(m/s). `None` preserves importer values. Dict maps joint regex→value |

**Unit conversion note**: For revolute joints, user-facing values (Nm/rad) are
automatically converted to USD convention (Nm/deg) by multiplying by `pi/180`.
Prismatic joint values are stored directly.

### Per-joint regex gains example

```python
joint_drive=UrdfConverterCfg.JointDriveCfg(
    drive_type={
        "arm_joint[1-7]": "acceleration",
        "finger": "force",
    },
    target_type="position",
    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
        stiffness={
            "arm_joint[1-7]": 100.0,
            "finger": 200.0,
        },
        damping={
            "arm_joint[1-7]": 10.0,
            "finger": 20.0,
        },
    ),
)
```

Regex patterns use `re.search` — partial matches work. If a pattern matches
no joints, a `ValueError` is raised listing available joint names.

## merge_fixed_joints details

When `merge_fixed_joints=True` (default), a URDF XML pre-processing step runs
**before** USD conversion:

1. Finds all `<joint type="fixed">` elements
2. For each fixed joint, merges the child link's `<visual>`, `<collision>`, and
   `<inertial>` elements into the parent link with proper transform composition
3. Re-parents any downstream joints that referenced the child link
4. Removes the fixed joint and child link elements
5. Iterates until no fixed joints remain (handles chains)

Inertial merge uses the **parallel axis theorem** to correctly combine mass,
center of mass, and inertia tensors.

The merged URDF is written next to the original file. If the source directory
is read-only, a temp directory is used as fallback (relative mesh paths may
not resolve in that case).

## Lazy conversion

The converter caches results via an MD5 hash of the config + URDF file
content. Re-conversion only happens when:

- The URDF file changes
- Configuration parameters change
- The output USD file doesn't exist
- `force_usd_conversion=True`

**Caveat**: Mesh file changes (STL/OBJ/DAE referenced by the URDF) do NOT
trigger re-conversion. Set `force_usd_conversion=True` or delete the output
directory if meshes change.

## Common pitfalls

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Using URDF importer without Isaac Sim | `ModuleNotFoundError: carb` / `omni.kit.app` | Install Isaac Sim: `./isaaclab.sh --install isaacsim` |
| Forgetting `fix_base=True` for fixed robots | Robot falls through the ground | Set `fix_base=True` |
| Setting `joint_drive=None` on articulations | No drive API on joints, robot goes limp | Use `JointDriveCfg()` with appropriate gains |
| Using `NaturalFrequencyGainsCfg` | Deprecated in importer 3.0, gains ignored | Use `PDGainsCfg` instead |
| Changing meshes but not URDF | Cached USD is stale | Set `force_usd_conversion=True` or delete output dir |
| `merge_fixed_joints=True` with read-only URDF dir | Warning + temp dir fallback, mesh paths may break | Copy URDF to a writable directory first |
| Setting `stiffness=None, damping=None` with actuators | Gains from URDF preserved, then overridden by actuator cfg | This is the **correct** pattern — let actuators control gains |
| Regex pattern matching no joints | `ValueError` at conversion time | Check joint names in URDF, use `.*` for all joints |

## Testing URDF imports

Run the existing test suite (requires Isaac Sim):

```bash
./isaaclab.sh -p -m pytest source/isaaclab/test/sim/test_urdf_converter.py
```

For a quick validation of a specific URDF:

```python
# Validate URDF parses correctly (no Isaac Sim needed)
import xml.etree.ElementTree as ET
tree = ET.parse("/path/to/robot.urdf")
root = tree.getroot()
links = root.findall("link")
joints = root.findall("joint")
print(f"Links: {len(links)}, Joints: {len(joints)}")
for j in joints:
    print(f"  {j.get('name')}: type={j.get('type')}, "
          f"parent={j.find('parent').get('link')}, "
          f"child={j.find('child').get('link')}")
```

To test merge_fixed_joints independently:

```python
from isaaclab.sim.converters.urdf_utils import merge_fixed_joints
merge_fixed_joints("/path/to/input.urdf", "/path/to/merged.urdf")
```

## Key source files

| File | Purpose |
|------|---------|
| `source/isaaclab/isaaclab/sim/converters/urdf_converter.py` | Main converter class |
| `source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py` | Configuration dataclass |
| `source/isaaclab/isaaclab/sim/converters/urdf_utils.py` | XML pre-processing (merge_fixed_joints) |
| `source/isaaclab/isaaclab/sim/converters/asset_converter_base.py` | Base class with lazy caching |
| `source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py` | `UrdfFileCfg` for use in asset configs |
| `source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py` | `spawn_from_urdf()` function |
| `source/isaaclab/test/sim/test_urdf_converter.py` | Test suite |

## Deprecated options (importer 3.0)

These options are no longer functional and will emit warnings:

- `convert_mimic_joints_to_normal_joints` — mimic joints handled natively
- `replace_cylinders_with_capsules` — no longer supported
- `root_link_name` — root link determined by PhysX
- `NaturalFrequencyGainsCfg` — use `PDGainsCfg` instead
