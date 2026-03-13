# Physics Backends, Renderers & Visualizers

## Architecture Overview

Isaac Lab 3.0 introduced a multi-backend architecture. The core `isaaclab` package
defines abstract interfaces; backend packages provide concrete implementations.

```
isaaclab.physics.PhysicsManager  (abstract)
  ├── isaaclab_physx.PhysxManager    (PhysX / Isaac Sim)
  └── isaaclab_newton.NewtonManager  (Newton / MuJoCo-Warp)

isaaclab.renderers.Renderer  (factory)
  ├── isaaclab_physx: Isaac RTX         (photorealistic, requires Isaac Sim)
  ├── isaaclab_newton: Newton Warp      (GPU-accelerated via Warp)
  └── isaaclab_ov: OVRTX               (Omniverse RTX)
```

The factory pattern (`FactoryBase` in `isaaclab.utils.backend_utils`) dynamically
selects the correct implementation at runtime based on the active physics backend.

## Physics Backends

### PhysX (`isaaclab_physx`)

- Full-featured physics via NVIDIA PhysX
- Supports: articulations, rigid objects, deformable objects, surface grippers
- Requires Isaac Sim runtime
- Location: `source/isaaclab_physx/`

### Newton (`isaaclab_newton`)

- GPU-accelerated physics via Newton (MuJoCo-Warp solver)
- Supports: articulations, rigid objects, rigid object collections
- **Kit-less** — does not require Isaac Sim
- Location: `source/isaaclab_newton/`

### Choosing a Backend

| Feature | PhysX | Newton |
|---------|-------|--------|
| Requires Isaac Sim | Yes | No |
| Articulations | Yes | Yes |
| Rigid objects | Yes | Yes |
| Deformable objects | Yes | No |
| Surface grippers | Yes | No |
| Kit-less install | No | Yes |

## Renderers

| Renderer | Backend Package | Description |
|----------|----------------|-------------|
| Isaac RTX | `isaaclab_physx` | Photorealistic RTX ray-tracing (requires Isaac Sim) |
| Newton Warp | `isaaclab_newton` | GPU-accelerated Warp-based rendering |
| OVRTX | `isaaclab_ov` | Omniverse RTX rendering |

## Visualizers

Visualizers provide real-time scene viewing. They are separate from renderers.

| Visualizer | Config | Description |
|-----------|--------|-------------|
| **Kit** | `KitVisualizerCfg` | Isaac Sim viewport (RTX, requires GUI/display) |
| **Newton** | `NewtonVisualizerCfg` | Native Newton physics renderer |
| **Rerun** | `RerunVisualizerCfg` | Browser-based (WebGL via gRPC), good for remote/distributed |
| **Viser** | `ViserVisualizerCfg` | Web-based viewer (Newton's ViewerViser), auto-opens browser |

Source: `source/isaaclab_visualizers/isaaclab_visualizers/`

All visualizers implement `BaseVisualizer` from `isaaclab.visualizers`:
- `initialize(scene_data_provider)`
- `step(sim_dt, scene_data_provider)`
- `close()`

### Visualizer CLI Usage

```bash
# Use Kit viewport (requires display)
./isaaclab.sh -p script.py --visualizer kit

# Use Newton renderer
./isaaclab.sh -p script.py --visualizer newton

# Use Rerun (browser)
./isaaclab.sh -p script.py --visualizer rerun

# Use Viser (browser)
./isaaclab.sh -p script.py --visualizer viser

# Headless (no visualization, max performance)
./isaaclab.sh -p script.py --headless
```

## Key Source Paths

| Component | Path |
|-----------|------|
| Physics base | `source/isaaclab/isaaclab/physics/physics_manager.py` |
| Factory utils | `source/isaaclab/isaaclab/utils/backend_utils.py` |
| PhysX manager | `source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py` |
| Newton manager | `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py` |
| Renderer factory | `source/isaaclab/isaaclab/renderers/renderer.py` |
| Visualizer factory | `source/isaaclab/isaaclab/visualizers/visualizer.py` |
| Visualizer impls | `source/isaaclab_visualizers/isaaclab_visualizers/` |

## Migration Reference

For details on migrating from Lab 2.x to 3.0:
```
docs/source/migration/migrating_to_isaaclab_3-0.rst
```
