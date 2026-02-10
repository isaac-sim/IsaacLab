# Newton Warp Renderer Integration Plan

## Goal
Integrate Newton Warp renderer to use Warp-based ray tracing for camera rendering while maintaining PhysX for physics simulation.

**Current**: PhysX simulation + RTX rendering  
**Target**: PhysX simulation + Newton Warp rendering

## Architecture Overview

### Current Stack (RTX Rendering)
```
┌─────────────────────────────────────┐
│         RL Training Loop            │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      TiledCamera Sensor             │
│    (Replicator/RTX Annotators)      │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│    Isaac Sim RTX Renderer           │
│    (Ray tracing on RTX cores)       │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│       PhysX Simulation              │
│     (Rigid body dynamics)           │
└─────────────────────────────────────┘
```

### Target Stack (Newton Warp Rendering)
```
┌─────────────────────────────────────┐
│         RL Training Loop            │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      TiledCamera Sensor             │
│   (renderer_type="newton_warp")     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│    Newton Warp Renderer             │
│  (SensorTiledCamera from Newton)    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│       Newton Model (Warp)           │
│   (Converts PhysX → Warp format)    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│       PhysX Simulation              │
│     (Rigid body dynamics)           │
└─────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Create Renderer Infrastructure

#### 1.1 Create Renderer Base Class
**File**: `source/isaaclab/isaaclab/renderer/__init__.py`
```python
from .renderer import RendererBase
from .newton_warp_renderer import NewtonWarpRenderer
from .newton_warp_renderer_cfg import NewtonWarpRendererCfg

__all__ = ["RendererBase", "NewtonWarpRenderer", "NewtonWarpRendererCfg"]

def get_renderer_class(renderer_type: str):
    """Get renderer class by type name."""
    if renderer_type == "newton_warp":
        return NewtonWarpRenderer
    return None
```

#### 1.2 Port Renderer Base Class
**File**: `source/isaaclab/isaaclab/renderer/renderer.py`

Create abstract base class for all renderers:
```python
class RendererBase:
    def __init__(self, cfg):
        self._width = cfg.width
        self._height = cfg.height
        self._num_envs = cfg.num_envs
        self._output_data_buffers = {}
    
    def initialize(self):
        raise NotImplementedError
    
    def render(self, positions, orientations, intrinsics):
        raise NotImplementedError
    
    def get_output(self) -> dict:
        return self._output_data_buffers
```

#### 1.3 Port Newton Warp Renderer
**File**: `source/isaaclab/isaaclab/renderer/newton_warp_renderer.py`

Source: [newton_warp_renderer.py](https://github.com/ooctipus/IsaacLab/blob/newton/dexsuite_warp_rendering/source/isaaclab/isaaclab/renderer/newton_warp_renderer.py)

Key components:
- `NewtonWarpRenderer` class
- Camera ray computation from intrinsics
- Warp kernels for format conversion
- Integration with `SensorTiledCamera` from Newton

#### 1.4 Create Renderer Config
**File**: `source/isaaclab/isaaclab/renderer/newton_warp_renderer_cfg.py`

```python
from dataclasses import dataclass

@dataclass
class NewtonWarpRendererCfg:
    width: int
    height: int
    num_cameras: int
    num_envs: int
```

### Step 2: Modify TiledCamera to Support Renderer Selection

#### 2.1 Update TiledCameraCfg
**File**: `source/isaaclab/isaaclab/sensors/camera/tiled_camera_cfg.py`

Add renderer type parameter:
```python
@configclass
class TiledCameraCfg(CameraCfg):
    renderer_type: str | None = None  # "newton_warp" or None (default RTX)
```

#### 2.2 Update TiledCamera Initialization
**File**: `source/isaaclab/isaaclab/sensors/camera/tiled_camera.py`

Source: [tiled_camera.py](https://github.com/ooctipus/IsaacLab/blob/newton/dexsuite_warp_rendering/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py)

Key changes in `_initialize_impl()`:
```python
if self.cfg.renderer_type == "newton_warp":
    renderer_cfg = NewtonWarpRendererCfg(
        width=self.cfg.width,
        height=self.cfg.height,
        num_cameras=self._view.count,
        num_envs=self._num_envs
    )
    renderer_cls = get_renderer_class("newton_warp")
    self._renderer = renderer_cls(renderer_cfg)
    self._renderer.initialize()
else:
    # Use default RTX rendering (existing code)
    ...
```

### Step 3: Create Newton Model Manager

#### 3.1 Create Newton Manager
**File**: `source/isaaclab/isaaclab/sim/_impl/newton_manager.py`

```python
class NewtonManager:
    """Manages Newton Warp model for rendering."""
    
    _model = None
    _state_0 = None
    
    @classmethod
    def initialize(cls, physics_context):
        """Initialize Newton model from PhysX context."""
        # Convert PhysX scene to Newton Warp model
        ...
    
    @classmethod
    def get_model(cls):
        return cls._model
    
    @classmethod
    def get_state_0(cls):
        return cls._state_0
    
    @classmethod
    def update_state(cls):
        """Update Newton state from PhysX."""
        ...
```

### Step 4: PhysX to Newton State Conversion

#### 4.1 State Conversion Layer
**Challenge**: Convert PhysX rigid body state to Newton Warp format

**Required conversions**:
- Rigid body positions/orientations → Warp arrays
- Mesh geometries → Newton model shapes
- Material properties → Newton material definitions
- Contact information (if needed)

**Approach**:
1. Initialize Newton model with same scene structure as PhysX
2. Each step, copy PhysX state tensors to Newton state arrays
3. Use Warp's efficient GPU-to-GPU copying

### Step 5: Update Vision Environment Config

#### 5.1 Modify Scene Config
**File**: `dexsuite_kuka_allegro_vision_env_cfg.py`

Update camera configuration:
```python
base_camera = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Camera",
    offset=TiledCameraCfg.OffsetCfg(...),
    data_types=MISSING,
    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 2.5)),
    width=MISSING,
    height=MISSING,
    renderer_type="newton_warp",  # NEW: Use Newton Warp renderer
)
```

### Step 6: Testing & Validation

#### 6.1 Unit Tests
- Test renderer initialization
- Test camera ray computation
- Test format conversions (RGB, depth)
- Test PhysX → Newton state conversion

#### 6.2 Integration Tests
- Single environment rendering
- Multi-environment (2048) rendering
- Compare rendering output RTX vs Newton
- Measure performance difference

#### 6.3 Training Validation
- Run training with Newton Warp renderer
- Verify observation shapes match
- Compare training convergence with RTX baseline
- Benchmark throughput (steps/s)

## Key Challenges & Solutions

### Challenge 1: PhysX → Newton Model Conversion
**Issue**: Newton expects its own model format, not PhysX state directly

**Solution Options**:
1. **Dynamic Conversion**: Each step, read PhysX state and update Newton model
2. **Shared Memory**: Use Warp arrays as backing storage for both PhysX and Newton
3. **Minimal Model**: Create simplified Newton model with only rendered objects

**Recommended**: Start with option 1 (dynamic conversion) for simplicity, optimize later

### Challenge 2: Performance Overhead
**Issue**: Converting PhysX → Newton may add latency

**Mitigation**:
- Use GPU-to-GPU copies (avoid CPU)
- Batch conversions for all environments
- Profile and optimize conversion kernels
- Consider caching static geometry

### Challenge 3: Material & Appearance
**Issue**: Newton may need separate material definitions

**Solution**:
- Define material mappings in scene config
- Convert PhysX materials to Newton format during initialization
- Use default appearance for prototyping

### Challenge 4: Coordinate Frame Conventions
**Issue**: PhysX and Newton may use different conventions

**Solution**:
- Already handled in renderer: `convert_camera_frame_orientation_convention()`
- Verify world frame alignment
- Add conversion utilities if needed

## File Structure

```
source/isaaclab/isaaclab/
├── renderer/
│   ├── __init__.py                    # NEW
│   ├── renderer.py                    # NEW: Base class
│   ├── newton_warp_renderer.py        # NEW: From Newton branch
│   └── newton_warp_renderer_cfg.py    # NEW
├── sim/
│   └── _impl/
│       └── newton_manager.py          # NEW: PhysX → Newton conversion
└── sensors/
    └── camera/
        ├── tiled_camera.py            # MODIFY: Add renderer selection
        └── tiled_camera_cfg.py        # MODIFY: Add renderer_type param
```

## Dependencies

### Required Packages
- `newton` - Warp-based physics simulator
- `warp` - Already installed
- Isaac Sim PhysX - Already available

### Import Additions
```python
# In newton_warp_renderer.py
from newton.sensors import SensorTiledCamera
from isaaclab.sim._impl.newton_manager import NewtonManager

# In tiled_camera.py  
from isaaclab.renderer import NewtonWarpRendererCfg, get_renderer_class
```

## Performance Goals

### Target Metrics
- **Initialization**: < 30s for 2048 environments
- **Rendering FPS**: > 60 FPS for 2048 environments
- **Training Throughput**: ≥ 1500 steps/s (match RTX baseline)
- **Memory**: < 20GB GPU memory

### Optimization Opportunities
1. Reuse camera rays (computed once from intrinsics)
2. Batch render all cameras in single kernel launch
3. Zero-copy data sharing where possible
4. Async rendering (render while physics steps)

## Testing Command

```bash
cd /home/perflab1/git/IsaacLab-Physx-Warp
conda activate physx_dextrah

# Test with Newton Warp renderer
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
  --enable_cameras \
  --num_envs=2048 \
  --max_iterations=32 \
  --logger=tensorboard \
  --headless \
  env.scene=64x64tiled_depth \
  env.scene.base_camera.renderer_type=newton_warp
```

## Success Criteria

1. ✅ Newton Warp renderer successfully initializes
2. ✅ Renders 2048 camera views without errors
3. ✅ Output format matches RTX renderer (shape, dtype)
4. ✅ Training loop runs end-to-end
5. ✅ Performance ≥ 80% of RTX baseline throughput
6. ✅ Visual output quality comparable to RTX

## Next Actions

1. **Port renderer infrastructure** (Step 1)
2. **Modify TiledCamera** (Step 2)
3. **Create Newton Manager** (Step 3)
4. **Test single environment** rendering
5. **Scale to multi-environment**
6. **Run full training benchmark**

## References

- [Newton Warp Renderer Source](https://github.com/ooctipus/IsaacLab/blob/newton/dexsuite_warp_rendering/source/isaaclab/isaaclab/renderer/newton_warp_renderer.py)
- [TiledCamera with Renderer Selection](https://github.com/ooctipus/IsaacLab/blob/newton/dexsuite_warp_rendering/source/isaaclab/isaaclab/sensors/camera/tiled_camera.py)
- Newton Warp Documentation
- Isaac Sim PhysX API
