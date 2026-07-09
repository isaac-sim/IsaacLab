# Event Randomization Examples

## Contents

- Reset state randomization
- Direct workflow event config
- Prestartup USD randomization
- Startup property randomization
- Backend-specific material randomization
- Interval disturbance

## Reset State Randomization

Input: randomize robot initial joint positions at the start of each episode.

Expected setup:

- Use a reset event.
- Scope the term to the robot articulation.
- Use conservative joint ranges first.
- Validate with repeated resets before training.

## Direct Workflow Event Config

Input: add friction and gravity randomization to a `DirectRLEnv` task.

Expected setup:

- Define an `EventCfg` with `EventTerm` entries in the direct task config module.
- Assign `events: EventCfg = EventCfg()` on the direct task config.
- Keep reward and observation logic in direct methods.
- Validate that `prestartup`, `startup`, `reset`, and `interval` modes fire at the expected times for direct workflows.

## Startup Property Randomization

Input: randomize a property once after simulation starts.

Expected setup:

- Use a startup event.
- Keep ranges tied to physical units.
- Confirm the randomized values are applied before rollout.

## Prestartup USD Randomization

Input: randomize authored USD-stage properties or asset variants before simulation buffers are created.

Expected setup:

- Use a prestartup event.
- Do not move the change to reset or interval events unless the backend explicitly supports changing that property after initialization.
- If per-episode variation is required but unsupported, pre-generate variants or use separate authored assets.

## Interval Disturbance

Input: apply random pushes during locomotion training.

Expected setup:

- Use an interval event.
- Choose a clear interval and magnitude range.
- Start with a small number of environments and inspect failures before scaling.

## Backend-Specific Material Randomization

Input: randomize rigid-body material properties in an environment that should support PhysX and Newton.

Expected setup:

- Read the `randomize_rigid_body_material` implementation before choosing parameters.
- Use PhysX buckets and static/dynamic friction ranges for the PhysX preset.
- Use Newton's single friction coefficient behavior for the Newton preset.
- Do not assume `dynamic_friction_range`, `num_buckets`, or CPU/GPU behavior are identical across backends.

Pattern:

```python
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg


@configclass
class PhysxEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )


@configclass
class NewtonEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 1,
        },
    )


@configclass
class EventCfg(PresetCfg):
    default = PhysxEventCfg()
    physx = PhysxEventCfg()
    newton_mjwarp = NewtonEventCfg()
```

Then assign `events: EventCfg = EventCfg()` on the environment config. Verify the exact event parameters against `source/isaaclab/isaaclab/envs/mdp/events.py` before using this pattern.
