# OVPhysX Manipulation Support — Design Spec

**Issue:** [#5322 — \[OVPHYSX\] Support for manipulation](https://github.com/isaac-sim/IsaacLab/issues/5322)
**Date:** 2026-04-20
**Status:** Draft

## Summary

Enable manipulation tasks to run on the OVPhysX backend. Like rough terrain locomotion, this is an integration task that validates core components work together for robotic manipulation.

**Target environments:**
- `Isaac-Repose-Cube-Allegro-Direct-v0` (in-hand reorientation — simplest, uses Articulation + RigidObject)
- `Isaac-Lift-Cube-Franka-v0` (cube lifting — uses Articulation + RigidObject + FrameTransformer)
- `Isaac-Stack-Cube-Franka-v0` (cube stacking — uses Articulation + RigidObjectCollection + FrameTransformer)

## Prerequisites

| Component | Issue | Required For |
|-----------|-------|--------------|
| Articulation | PR #4852 (done) | Robot arm control |
| RigidObject | #5316 | Manipulated objects |
| RigidObjectCollection | #5317 | Multi-object tasks (stacking) |
| FrameTransformer | #5320 | End-effector tracking |
| ContactSensor | New issue | Grasp detection (DexSuite) |
| SceneDataProvider | New issue | Visualization |

**Staged approach:**
1. **Allegro hand repose** — needs Articulation + RigidObject only (fewest deps)
2. **Franka cube lift** — adds FrameTransformer
3. **Franka cube stack** — adds RigidObjectCollection

## What Needs to Work

1. **Robot arm control** — Joint position targets for Franka/Allegro
2. **Object interaction** — RigidObject responds to contact forces from gripper
3. **Frame tracking** — FrameTransformer reports end-effector pose relative to base
4. **Object state observations** — Position, orientation, velocity of manipulated objects
5. **Episode resets** — Reset both robot and object to initial states
6. **Gripper control** — Open/close for Franka parallel jaw (action space)

### Potential Blockers

**Object-robot contact:** Verify ovphysx handles articulation-rigid body contact correctly. The DexCube must respond to forces from the Allegro hand or Franka gripper.

**Gripper joint limits:** Franka gripper has prismatic joints with force limits. Verify ovphysx joint limit and force clamping works.

## Implementation Steps

1. **Allegro hand repose** (first — validates RigidObject):
   - Add OVPhysX config variant
   - Run for 100 steps, verify cube falls and hand contacts it
   - Verify resets work

2. **Franka cube lift** (second — validates FrameTransformer):
   - Add OVPhysX config variant
   - Verify EE frame transformer outputs sensible relative poses
   - Verify cube pickup physics

3. **Franka cube stack** (third — validates RigidObjectCollection):
   - Add OVPhysX config variant
   - Verify multi-object state tracking
   - Verify independent object resets

## Task Configuration Changes

```python
@configclass
class FrankaLiftOvPhysxEnvCfg(FrankaCubeLiftEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        physics_cfg=OvPhysxCfg(...)
    )
```

## Validation

```bash
# Stage 1: Allegro hand
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/allegro_hand/allegro_hand_env.py \
    --num_envs 4

# Stage 2: Franka lift (manager-based, needs rsl_rl or similar)
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_environments.py \
    -k "Isaac-Lift-Cube-Franka-v0"

# Stage 3: Franka stack
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_environments.py \
    -k "Isaac-Stack-Cube-Franka-v0"
```

### Training validation
- Run 10K steps of Allegro hand repose training
- Verify reward signal and learning

## Dependencies

- **Requires #5316 (RigidObject)** for stage 1
- **Requires #5320 (FrameTransformer)** for stage 2
- **Requires #5317 (RigidObjectCollection)** for stage 3
- **Requires PR #4852 merged** (Articulation)

## Estimated Scope

- OVPhysX task config files: ~50 lines each (3 configs)
- Integration bug fixes: variable
- No new core implementation
