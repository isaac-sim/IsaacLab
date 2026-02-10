# IsaacLab Physx-Warp Setup Guide

## Overview
This guide documents the setup and fixes required to run vision-based dexterous manipulation tasks in IsaacLab with PhysX simulation and RTX rendering.

## Environment Setup

### Python Environment
- **Python Version**: 3.12 (required for Isaac Sim 6.0 compatibility)
- **Conda Environment**: `physx_dextrah`
- **Key Change**: Updated from Python 3.11 to 3.12 to match compiled Isaac Sim bindings

### Environment Configuration Files Modified

1. **environment.yml**
   - Changed: `python=3.11` → `python=3.12`
   - Location: `/home/perflab1/git/IsaacLab-Physx-Warp/environment.yml`

2. **Conda Environment Update**
   ```bash
   conda activate physx_dextrah
   conda install python=3.12
   ```

### Python Package Fixes

#### 1. flatdict Package (Python 3.10+ Compatibility)
**Issue**: `collections.MutableMapping` removed in Python 3.10+

**Location**: `/home/perflab1/miniconda3/envs/physx_dextrah/lib/python3.12/site-packages/flatdict.py`

**Changes**:
- Line 5: `import collections` → `import collections.abc`
- Line 18: `class FlatDict(collections.MutableMapping)` → `class FlatDict(collections.abc.MutableMapping)`

**Alternative Fix**: Updated `source/isaaclab/setup.py` to allow `flatdict>=3.4.0` instead of strict `==4.0.1`

## Vision Task Implementation

### Files Created/Modified

#### 1. Vision Environment Configuration
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/kuka_allegro/dexsuite_kuka_allegro_vision_env_cfg.py`

**Source**: Copied from Newton-Warp repository
- Defines camera scene configurations for single/dual camera setups
- Configures TiledCamera with 64x64 resolution
- Sets up vision-based observation groups

#### 2. Base Environment Configuration Updates
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py`

**Added**:
- `FINGERTIP_LIST` constant for finger link names
- `KukaAllegroSceneCfg` class - defines robot scene with contact sensors
- `KukaAllegroObservationCfg` class - configures proprio observations including contact forces

#### 3. Vision Camera Observation Term
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/mdp/observations.py`

**Added**:
- Import: `from isaaclab.sensors import TiledCamera`
- `vision_camera` class - Manager term for retrieving and normalizing camera data
  - Supports RGB and depth normalization
  - Handles NaN values in camera output
  - Permutes dimensions for CNN compatibility (NHWC → NCHW)

**Fixed**:
- `fingers_contact_force_b` function - added `.view(env.num_envs, -1)` to flatten output for proper concatenation with other observations

#### 4. Task Registration
**File**: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/kuka_allegro/__init__.py`

**Added Gym Registrations**:
- `Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0`
- `Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-Play-v0`
- `Isaac-Dexsuite-Kuka-Allegro-Reorient-Single-Camera-v0`
- `Isaac-Dexsuite-Kuka-Allegro-Reorient-Single-Camera-Play-v0`

## Running Vision-Based Training

### Command
```bash
cd /home/perflab1/git/IsaacLab-Physx-Warp
source /home/perflab1/miniconda3/etc/profile.d/conda.sh
conda activate physx_dextrah
export WANDB_USERNAME=perflab1

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0 \
  --enable_cameras \
  --num_envs=2048 \
  --max_iterations=32 \
  --logger=tensorboard \
  --headless \
  env.scene=64x64tiled_depth
```

### Performance Metrics
- **Throughput**: ~1675-1737 steps/s with 2048 environments
- **Observation Shape**: 
  - `policy`: (34,) - object pose, target, actions
  - `proprio`: (123,) - contacts, joint states, hand tips
  - `perception`: (192,) - object point cloud
  - `base_image`: (1, 64, 64) - camera depth/RGB

## Key Issues Resolved

### 1. Python Version Mismatch
**Error**: `TypeError: 'NoneType' object is not callable` for SimulationApp
**Root Cause**: Conda environment using Python 3.11, Isaac Sim compiled for 3.12
**Solution**: Updated conda environment to Python 3.12

### 2. Missing Vision Camera Term
**Error**: `AttributeError: module 'mdp' has no attribute 'vision_camera'`
**Root Cause**: Vision observation term not implemented in Physx-Warp branch
**Solution**: Ported `vision_camera` class from Newton-Warp branch

### 3. Observation Shape Mismatch
**Error**: `RuntimeError: Unable to concatenate observation terms... shapes [(4, 3), (23,), ...]`
**Root Cause**: `fingers_contact_force_b` returning 2D tensor instead of flattened
**Solution**: Added `.view(env.num_envs, -1)` to flatten the output

### 4. Missing Scene and Observation Configs
**Error**: `AttributeError: module has no attribute 'KukaAllegroSceneCfg'`
**Root Cause**: Newton-Warp uses different config structure than Physx-Warp
**Solution**: Created missing config classes in base environment file

## Current Architecture

### Simulation & Rendering Stack
- **Physics**: PhysX (Isaac Sim 6.0)
- **Rendering**: RTX (Replicator/native Isaac Sim)
- **Camera**: TiledCamera using RTX rendering pipeline
- **Framework**: IsaacLab with ManagerBasedRLEnv

### Observation Pipeline
1. TiledCamera renders 2048 environments using RTX
2. `vision_camera` term retrieves and normalizes images
3. Images concatenated with proprio/policy observations
4. Fed to policy network (MLP: 349 → 512 → 256 → 128 → 23)

## Next Steps: Newton Warp Renderer Integration

### Goal
Replace RTX rendering with Warp-based rendering while keeping PhysX simulation:
- **Simulation**: PhysX (current)
- **Rendering**: Newton Warp Renderer (new)
- **Data Flow**: PhysX state → Newton model → Warp ray tracing → rendered output

### Key Files to Integrate
1. `isaaclab/renderer/newton_warp_renderer.py` - Warp rendering backend
2. `isaaclab/sensors/camera/tiled_camera.py` - Camera interface with renderer selection

### Implementation Strategy
1. Create new branch for Newton Warp integration
2. Port Newton Warp renderer to Physx-Warp branch
3. Modify TiledCamera to support `renderer_type="newton_warp"` parameter
4. Configure vision tasks to use Newton renderer via scene config
5. Ensure PhysX state properly converts to Newton model format

## References
- Isaac Sim Version: 6.0 (Kit 110.0.0)
- IsaacLab: Physx-Warp branch
- Newton Warp Renderer: [GitHub](https://github.com/ooctipus/IsaacLab/blob/newton/dexsuite_warp_rendering/source/isaaclab/isaaclab/renderer/newton_warp_renderer.py)
