# Soft-body grasping study

Deformable-object grasping experiments on `Isaac-Lift-Soft-Franka-v0` (Newton MJWarp + VBD,
and PhysX FEM), plus the analysis tooling and notes that go with them.

Branched from `v3.0.0-beta2.patch1`. **Not** based on `main`: the `lift_franka_soft` task and
`Isaac-Lift-Soft-Franka-v0` do not exist on upstream `main` — they only ship in the 3.0 beta2
release line, and `v3.0.0-beta2.patch1` is not an ancestor of `main`. Everything here depends
on that task, so it must stay on the beta2 line until the task lands upstream.

## Layout

| Path | What |
|---|---|
| `docs/HANDOFF-ubuntu-curobo-pick-place.md` | **Start here.** Findings, Ubuntu migration, cuRobo install, pick-and-place design |
| `docs/IsaacLab-SoftLift-Newton-vs-PhysX.md` | Earlier Newton vs PhysX benchmark notes (Windows) |
| `analysis/` | Plot scripts and raw sweep results (CSV) |
| `assets/make_strawberry_usd.py` | YCB scan → watertight → decimated → `UsdGeom.Mesh` USD |

The generated `strawberry_deformable.usda` is **not committed** — IsaacLab's `.gitignore`
excludes `**/*.usda`. Regenerate it by downloading the YCB scan and running
`make_strawberry_usd.py` (the command is in its docstring).

Simulation scripts live with the stock example in `scripts/environments/state_machine/`:

| Script | What |
|---|---|
| `lift_franka_soft_verify.py` | Bounded self-check of the stock task |
| `lift_franka_soft_compare.py` | Newton vs PhysX + video capture |
| `lift_franka_soft_physx_tune.py` | Grasp-force / state-machine tuning |
| `bench_one.py` | Single-point throughput + VRAM benchmark |
| `grasp_determinism.py` | Repeatability: constant EE pose, gripper-only actuation |
| `grasp_sweep.py` | Mesh / stiffness / force sweeps, with a tet-mesh disk cache |
| `grasp_strawberry.py` | Strawberry deformable grasp + Newton render |

## Headline results

- **The tet mesh is non-deterministic.** `pytetwild` (fTetWild) is randomised and unseeded, so the
  same cuboid tetrahedralises to 61–74 nodes across launches. Same command, same seed, same force
  gave grasped widths of 62.06 mm vs 40.28 mm in two processes — a 35 % swing.
  `grasp_sweep.py` pins it with a disk cache; the proper fix is shipping a pre-built `UsdGeom.TetMesh`.
- **Mesh resolution dominates the material.** 39.9 mm spread on a 50 mm object across 13–533 nodes,
  non-monotonic and not converged — larger than a 16× change in Young's modulus.
- **Hooke's law holds.** Gap is linear in `1/E` (R² = 0.997) and in `F` (R² = 0.986), but ~4.5–5.4×
  more compliant than analytic `FL/EA`, because the finger pads cover only part of the face.
- **Finger jitter was a limit cycle, not noise.** A force-saturated drive emits constant force with
  its damping term clipped, so nothing removes energy. Switching to position-hold cut jitter ~130×
  (tail p2p 15.98 mm → 0.118 mm).

See the handoff doc for the full picture, including what did *not* work and why.

## Note on the asset root

`apps/isaaclab.python.kit` is pinned to `Assets/Isaac/5.0`. NVIDIA renamed
`Robots/FrankaEmika/panda_instanceable.usd` → `franka_panda.usda` in the 6.0 tree after beta2
shipped, so the stock 6.0 path 404s and every run dies with `FileNotFoundError`. Re-check before
keeping this — if 6.0 serves the asset again, drop that commit.
