# CORE benchmark coverage

These snapshots were queried directly from `omni_runtime_isaac_lab_v3` on September 10, 2026.
They include every positive WARM runtime/training result on the selected dates for
one RTX PRO 6000 Blackwell Server Edition GPU, at each task’s configured training-default
environment count. The same count applies to Collection and Training. No other count is used as a fallback.
No release measurements were inferred from develop or from older runs.

Physics: **P** = Isaac Sim PhysX, **N** = Newton MJWarp, **V** = Newton MJWarp + VBD,
**O** = OV PhysX. Renderers: **I** = Isaac Sim RTX, **W** = Newton Renderer, **R** = OV RTX.
For example, `N/R` means Newton MJWarp with OV RTX. A dash means no configuration gaps.

Expected configurations use the browser’s CORE task selectors, excluding Kamino and
the incompatible P/R and O/I pairs. Camera rows with an unspecified renderer do not
count as coverage for an explicit renderer. Coverage means at least one measured point;
it does not mean every date or every domain preset is present.

`Isaac-Fourbar-Pole-Swingup` is omitted because it only supports Kamino.
`Isaac-Pendulum-MARL-Direct` has no RSL-RL entry point; populating its training data
requires a benchmark from one of its supported libraries (RL-Games or SKRL).

## Release

| CORE task | Default envs | Collection gaps | Training gaps |
| --- | ---: | --- | --- |
| Isaac-Ant-Direct | 4096 | P, N, O | O |
| Isaac-Ant | 4096 | P, N, O | O |
| Isaac-Cartpole-Direct | 4096 | P, N, O | O |
| Isaac-Cartpole | 4096 | P, N, O | O |
| Isaac-Cartpole-Camera-Direct | 512 | P/I, P/W, N/I, N/W, N/R, O/W, O/R | P/I, P/W, N/I, N/W, N/R, O/W, O/R |
| Isaac-Cartpole-Camera | 512 | P/I, P/W, N/I, N/W, N/R, O/W, O/R | P/I, P/W, N/I, N/W, N/R, O/W, O/R |
| Isaac-Humanoid-Direct | 4096 | P, N, O | O |
| Isaac-Humanoid | 4096 | P, N, O | O |
| Isaac-Lift-Cable-Franka | 8192 | V | V |
| Isaac-Lift-Cable-Franka-Camera | 128 | V/I, V/W, V/R | V/I, V/W, V/R |
| Isaac-Lift-Cloth-Franka | 2048 | P, V | P, V |
| Isaac-Lift-Cloth-Franka-Camera | 128 | P/I, P/W, V/I, V/W, V/R | P/I, P/W, V/I, V/W, V/R |
| Isaac-Lift-Franka | 4096 | P, N, O | O |
| Isaac-Lift-KukaAllegro | 4096 | P, N, O | O |
| Isaac-Lift-KukaAllegro-Camera | 4096 | P/I, P/W, N/I, N/W, N/R, O/W, O/R | P/I, P/W, N/I, N/W, N/R, O/W, O/R |
| Isaac-Lift-Soft-Franka | 2048 | P, V | P, V |
| Isaac-Lift-Soft-Franka-Camera | 128 | P/I, P/W, V/I, V/W, V/R | P/I, P/W, V/I, V/W, V/R |
| Isaac-Open-Drawer-Franka-Direct | 4096 | P, N, O | O |
| Isaac-Open-Drawer-Franka | 4096 | P, N, O | O |
| Isaac-Pendulum-MARL-Direct | 4096 | P, N, O | P, N, O |
| Isaac-Reach-Franka | 4096 | P, N, O | O |
| Isaac-Reach-Franka-OSC | 4096 | P, N, O | O |
| Isaac-Reach-UR10 | 4096 | P, N, O | O |
| Isaac-Reorient-Cube-Allegro-Direct | 8192 | — | — |
| Isaac-Reorient-Cube-Allegro | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow-Direct | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 1225 | P/I, P/W, N/I, N/W, N/R, O/W, O/R | P/I, P/W, N/I, N/W, N/R, O/W, O/R |
| Isaac-Reorient-Cube-Shadow-Camera | 1225 | P/I, P/W, N/I, N/W, N/R, O/W, O/R | P/I, P/W, N/I, N/W, N/R, O/W, O/R |
| Isaac-Reorient-Franka | 4096 | P, N, O | O |
| Isaac-Reorient-KukaAllegro | 4096 | P, N, O | O |
| Isaac-Reorient-KukaAllegro-Camera | 4096 | P/I, P/W, N/I, N/W, N/R, O/W, O/R | P/I, P/W, N/I, N/W, N/R, O/W, O/R |
| Isaac-Shadow-Handover-Direct | 2048 | P, N, O | O |
| Isaac-Shadow-Handover | 2048 | P, N, O | P, N, O |
| Isaac-Velocity-Flat-AnymalD | 4096 | P, N, O | O |
| Isaac-Velocity-Flat-Cassie | 4096 | P, N, O | O |
| Isaac-Velocity-Flat-G1 | 4096 | P, N, O | O |
| Isaac-Velocity-Flat-H1 | 4096 | P, N, O | O |
| Isaac-Velocity-Flat-UnitreeGo2 | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-AnymalD | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-Cassie | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-G1 | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-H1 | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-UnitreeGo2 | 4096 | P, N, O | O |

## Develop

| CORE task | Default envs | Collection gaps | Training gaps |
| --- | ---: | --- | --- |
| Isaac-Ant-Direct | 4096 | P, N, O | O |
| Isaac-Ant | 4096 | P, N, O | O |
| Isaac-Cartpole-Direct | 4096 | P, N, O | O |
| Isaac-Cartpole | 4096 | P, N, O | O |
| Isaac-Cartpole-Camera-Direct | 512 | P/I, N/I, N/W, N/R, O/W, O/R | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Cartpole-Camera | 512 | P/I, N/I, N/W, N/R, O/W, O/R | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Humanoid-Direct | 4096 | P, N, O | O |
| Isaac-Humanoid | 4096 | P, N, O | O |
| Isaac-Lift-Cable-Franka | 8192 | V | V |
| Isaac-Lift-Cable-Franka-Camera | 128 | V/I, V/W, V/R | V/I, V/W, V/R |
| Isaac-Lift-Cloth-Franka | 2048 | P, V | P, V |
| Isaac-Lift-Cloth-Franka-Camera | 128 | P/I, P/W, V/I, V/W, V/R | P/I, P/W, V/I, V/W, V/R |
| Isaac-Lift-Franka | 4096 | P, N, O | O |
| Isaac-Lift-KukaAllegro | 4096 | P, N, O | O |
| Isaac-Lift-KukaAllegro-Camera | 4096 | P/I, N/I, N/W, N/R, O/W, O/R | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Lift-Soft-Franka | 2048 | P, V | P, V |
| Isaac-Lift-Soft-Franka-Camera | 128 | P/I, P/W, V/I, V/W, V/R | P/I, P/W, V/I, V/W, V/R |
| Isaac-Open-Drawer-Franka-Direct | 4096 | P, N, O | O |
| Isaac-Open-Drawer-Franka | 4096 | P, N, O | O |
| Isaac-Pendulum-MARL-Direct | 4096 | P, N, O | P, N, O |
| Isaac-Reach-Franka | 4096 | P, N, O | O |
| Isaac-Reach-Franka-OSC | 4096 | P, N, O | O |
| Isaac-Reach-UR10 | 4096 | P, N, O | O |
| Isaac-Reorient-Cube-Allegro-Direct | 8192 | — | — |
| Isaac-Reorient-Cube-Allegro | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow-Direct | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow | 8192 | — | — |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | 1225 | P/I, N/I, N/W, N/R, O/W, O/R | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Reorient-Cube-Shadow-Camera | 1225 | P/I, N/I, N/W, N/R, O/W, O/R | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Reorient-Franka | 4096 | P, N, O | O |
| Isaac-Reorient-KukaAllegro | 4096 | P, N, O | O |
| Isaac-Reorient-KukaAllegro-Camera | 4096 | P/I, N/I, N/W, N/R, O/W, O/R | P/I, N/I, N/W, N/R, O/W, O/R |
| Isaac-Shadow-Handover-Direct | 2048 | P, N, O | O |
| Isaac-Shadow-Handover | 2048 | P, N, O | P, N, O |
| Isaac-Velocity-Flat-AnymalD | 4096 | P, N, O | O |
| Isaac-Velocity-Flat-Cassie | 4096 | P, N, O | O |
| Isaac-Velocity-Flat-G1 | 4096 | P, N, O | O |
| Isaac-Velocity-Flat-H1 | 4096 | P, N, O | O |
| Isaac-Velocity-Flat-UnitreeGo2 | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-AnymalD | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-Cassie | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-G1 | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-H1 | 4096 | P, N, O | O |
| Isaac-Velocity-Rough-UnitreeGo2 | 4096 | P, N, O | O |

## Database findings

Release contains 76 measurements (12 Collection, 64 Training) across 30 tasks.
Develop contains 225 measurements (43 Collection, 182 Training) across 36 tasks.
Release has no camera measurements at task defaults. Develop retains only the
camera configurations with measurements at the default count, as listed above.

A broader search across all machines and all ingestion dates from August 17 through
September 9 also found no WARM benchmark entries for the eight entirely absent tasks:
the cable, cloth, and soft Franka tasks and their camera variants,
`Isaac-Pendulum-MARL-Direct`, and `Isaac-Shadow-Handover`.

These are gaps in the queried database window, not a claim about all historical data.

Release uses September 9; develop uses August 17, August 24, August 31, and September 7.

A curve contains only its measured dates; there is no interpolation or backfill
between channels. In particular, August 17 has fewer configurations than later dates.
