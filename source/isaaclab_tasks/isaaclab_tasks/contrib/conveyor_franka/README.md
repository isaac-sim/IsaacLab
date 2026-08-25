<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# Conveyor Franka

This package provides checkpoint-compatible Newton and PhysX variants of a manager-based task in
which a Franka transfers four numbered cubes between two counter-rotating racetrack conveyors.
Both variants preserve the same ordered eight-dimensional action space, policy observations,
commands, rewards, reset recipes, 120 Hz physics step, and 60 Hz policy rate.

## Backend support

| Task | Physics device | Intended use | Conveyor actuation |
| --- | --- | --- | --- |
| `IsaacContrib-Conveyor-Franka-Newton-v0` | CUDA | Training and scalable playback | Batched Warp contact-force feedback captured with the Newton solver graph |
| `IsaacContrib-Conveyor-Franka-Newton-Play-v0` | CUDA | Warehouse-dressed Digital Twin playback | Same Newton force feedback on lightweight hidden collision surfaces |
| `IsaacContrib-Conveyor-Franka-PhysX-CPU-v0` | CPU only | Native-PhysX reference and checkpoint playback | Authored `PhysxSurfaceVelocityAPI` on kinematic belt sections |

The PhysX task rejects CUDA during configuration validation. In the supported Isaac Sim runtime,
enabling the native surface-velocity contact-modification path under GPU dynamics can drop the belt
contacts and let packages pass through the conveyor. CPU PhysX preserves those contacts. Use the
Newton task whenever GPU simulation or vectorized throughput is required.

The two backends deliberately share the policy tensor contract, so an RSL-RL checkpoint can be
loaded by either task without reshaping or reordering tensors. Their contact and actuator dynamics
are not numerically identical; validate task behavior when transferring a policy between them.

## Newton GPU playback

Newton is kitless and supports the lightweight GL viewer:

```bash
DISPLAY=:1 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Franka-Newton-v0 \
  --checkpoint pretrained \
  --num_envs 8 --device cuda:0 --viz newton_gl --real-time
```

The `pretrained` selector downloads the RSL-RL policy published specifically for the Newton MJWarp
backend. To evaluate another policy, replace `pretrained` with an explicit checkpoint path. The
PhysX task resolves a different backend-specific artifact name, so transferring this Newton policy
to PhysX currently requires the explicit local checkpoint path shown below.

For presentation, use the checkpoint-compatible Play variant. It replaces the procedural render
geometry with `ConveyorBelt_A09` straight sections and `ConveyorBelt_A12` 180-degree turns from
`Isaac/Props/Conveyors`. A Thor robot table, packing station, separate loaded and empty pallet bays,
safety markings, and a warehouse backdrop complete the scene. Every added USD is render-only with its
authored physics APIs and action graphs stripped locally. The same lightweight hidden surfaces remain
the sole owners of contact and conveyor forces, so scene dressing cannot introduce double contacts or
change the trained policy dynamics. Measured asset feet sit on the global `z=0` ground plane; the complete
policy workspace is elevated without changing its local coordinates. This asset-rich variant defaults to
one environment to keep interactive startup and rendering practical.

```bash
DISPLAY=:1 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Franka-Newton-Play-v0 \
  --checkpoint pretrained \
  --num_envs 1 --device cuda:0 --viz newton_gl --real-time
```

Training uses the same task ID and defaults to 256 environments:

```bash
uv run isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Franka-Newton-v0 \
  --num_envs 256 --device cuda:0
```

## PhysX CPU playback

The native PhysX variant requires an Isaac Sim-enabled launch and an explicit CPU device. One
environment is the default and recommended interactive configuration:

```bash
DISPLAY=:1 uv run isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Conveyor-Franka-PhysX-CPU-v0 \
  --checkpoint /path/to/model.pt \
  --num_envs 1 --device cpu --viz kit --real-time \
  agent.device=cpu
```

Overriding the task to CUDA is an error by design; keep the explicit `--device cpu` in launch
commands for clarity. The native surface-velocity backend stages commands through USD, while the
Newton backend keeps its batched state, contact processing, and force application on the GPU.
