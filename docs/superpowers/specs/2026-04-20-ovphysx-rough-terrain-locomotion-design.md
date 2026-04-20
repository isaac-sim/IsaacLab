# OVPhysX Rough Terrain Locomotion Support — Design Spec

**Issue:** [#5321 — \[OVPHYSX\] Support for rough terrain locomotion](https://github.com/isaac-sim/IsaacLab/issues/5321)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Enable rough terrain locomotion tasks to run on the OVPhysX backend. This is an integration task, not a new implementation — it validates that the core backend components (Articulation, ContactSensor, terrain generation) work together correctly for legged locomotion.

**Target environments:**
- `Isaac-Velocity-Flat-Anymal-C-Direct-v0` (flat terrain, simpler)
- `Isaac-Velocity-Rough-Anymal-C-Direct-v0` (rough terrain, full validation)
- Manager-based velocity tracking tasks with various robots (Anymal, Unitree, etc.)

## Prerequisites

This task depends on several other OVPhysX components being complete:

| Component | Issue | Status | Required For |
|-----------|-------|--------|--------------|
| Articulation | PR #4852 | Done | Robot control |
| ContactSensor | New issue | Needed | Foot contact detection |
| RigidObject | #5316 | Needed | Terrain objects (if any) |
| SceneDataProvider | New issue | Nice-to-have | Visualization |

**ContactSensor is the critical dependency.** Without it, locomotion tasks cannot detect foot contacts for gait rewards.

## What Needs to Work

### Flat terrain (minimum viable)

1. **Robot spawning and cloning** — Anymal C articulation across N environments
2. **Joint control** — Position/velocity targets via actuators
3. **Contact detection** — Foot contact forces for gait rewards
4. **State observations** — Joint positions/velocities, root state, projected gravity
5. **Episode resets** — Root state write-back on termination

### Rough terrain (full validation)

All of the above plus:

6. **Terrain generation** — Height field or trimesh terrain
7. **Terrain-aware spawning** — Robots placed on valid terrain locations
8. **Height scanning** — Ray-cast or height-field queries for terrain observations

### Potential Blockers

**Terrain mesh loading:** OVPhysX loads from USD. If terrain is generated procedurally (height field → USD mesh), verify the pipeline works without Kit:
- `isaaclab.terrains` generates meshes and writes them to USD
- OVPhysX must load these meshes from the exported USD

**Scene queries (ray casts):** Some locomotion tasks use height scanning via ray casts. If ovphysx lacks scene query APIs, this feature is blocked. Check with @marcodiiga.

**Question for @marcodiiga:** Does ovphysx support:
1. Loading trimesh collision geometry from USD?
2. GPU scene queries (ray casts)?

## Implementation Steps

1. **Start with flat terrain** — verify Anymal C walks on flat ground
2. **Add terrain meshes** — verify procedural terrain loads in ovphysx
3. **Validate full rough terrain** — run the complete velocity tracking task
4. **Profile performance** — compare step time vs PhysX backend

## Task Configuration Changes

Create OVPhysX-specific task configs (or verify existing ones work):

```python
# Flat terrain with OVPhysX
@configclass
class AnymalCFlatOvPhysxEnvCfg(AnymalCFlatEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        physics_cfg=OvPhysxCfg(...)
    )
```

The environment logic itself should not change — only the physics backend config.

## Validation

### Flat terrain validation
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env.py \
    --num_envs 4 --task Isaac-Velocity-Flat-Anymal-C-Direct-v0
```

### Rough terrain validation
```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env.py \
    --num_envs 4 --task Isaac-Velocity-Rough-Anymal-C-Direct-v0
```

### Training validation
Run 10K steps of training and verify:
- Reward increases over time (learning signal exists)
- No NaN/inf in observations or rewards
- Episode resets work correctly
- Performance within 2x of PhysX for equivalent env count

## Dependencies

- **Requires ContactSensor** (blocks gait rewards)
- **Requires Articulation** (PR #4852, done)
- **May require scene query API** from ovphysx for height scanning
- **May require terrain mesh loading** verification

## Estimated Scope

- OVPhysX task config files: ~50 lines each
- Bug fixes in OVPhysX components: variable (discovered during integration)
- No new core implementation — this is an integration/validation task
