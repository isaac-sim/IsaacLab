# Handoff: move to Ubuntu, add cuRobo planning, build a fruit→pot pick-and-place

> Written 2026-08-06, on the Windows machine (`kuorobot02`, RTX 6000 Ada 48 GB, driver 596.86 / CUDA 13.2).
> Supersedes the Windows-specific parts of `IsaacLab-SoftLift-Newton-vs-PhysX.md`.
> Audience: whoever continues this on the Ubuntu server.

---

## 0. TL;DR

- The soft-body grasping work is **done and validated on Windows**. Results are in §3.
- The **pick-and-place + cuRobo task was not started** — it is blocked on Windows and deliberately deferred. Design is in §9.
- Move to Ubuntu. There are **four independent Windows-specific blockers** (§1), and Ubuntu removes all of them at once.
- Nothing here is a git repo (§6), so nothing was pushed. Read §6 before trying.

---

## 1. Why leave Windows

Not preference — four concrete, separately-verified blockers.

| # | Blocker | Evidence | Ubuntu fixes it? |
|---|---|---|---|
| 1 | **cuRobo cannot build.** Needs CUDA `nvcc` + a host C++ compiler to build its CUDA kernels. This machine has neither `nvcc` nor MSVC (no Visual Studio at all). Windows support is officially *experimental*. | `nvcc`/`cl.exe`/`vswhere` all absent | Yes — `apt install` gcc + CUDA toolkit |
| 2 | **Newton CUDA-graph capture fails.** Logs `libcudart not available` → falls back to eager. This is the documented root cause of Newton's poor scaling here (peak ~1.1k env-steps/s vs PhysX ~74k). | prior doc §4.3 | Very likely — it's a Windows-specific CUDA/driver issue |
| 3 | **Newton headless video needs EGL**, which is Linux-only. On Windows we had to monkeypatch `newton.viewer.ViewerGL` into windowed WGL mode just to capture frames. | `grasp_strawberry.py`, `lift_franka_soft_compare.py` | Yes — EGL headless works natively |
| 4 | **MAX_PATH DLL loading.** Isaac Sim ships `.pyd` files up to 193 chars below the venv root; Kit's loader is bound by MAX_PATH (259), so the venv root must be ≤66 chars. Broke 14 extensions incl. the entire PhysX stack until the project was moved to `C:\isaac_soft`. `LongPathsEnabled=1` does *not* help; `subst` does not either (Kit canonicalises). | §3.6 | Yes — no MAX_PATH on Linux |

Blocker 1 is the hard one for this task. Blocker 2 also means **any Newton performance number measured on Windows is not representative** — worth re-running the benchmarks on Ubuntu.

---

## 2. Current environment (what to reproduce)

| Component | Version | Note |
|---|---|---|
| Python | 3.12.13 | forced by Isaac Sim 6.0; **conflicts with cuRobo's tested 3.8–3.10**, see §8.3 |
| torch | 2.11.0+cu128 | Isaac Sim hard-pins 2.11.0 |
| torchvision / torchaudio | 0.26.0+cu128 / 2.11.0+cu128 | |
| isaacsim | 6.0.0.1 | `[all,extscache]` from pypi.nvidia.com |
| isaaclab / _tasks / _newton / _physx / _rl | 6.1.11 / 1.10.9 / 0.13.6 / 1.1.3 / 0.5.4 | editable installs from source |
| warp-lang | 1.13.0 | |
| Venv | `C:\isaac_soft\kuorobot02_python` (~15 GB) | per-machine name; do **not** copy it to Ubuntu, rebuild |

---

## 3. What has been established (do not re-derive)

### 3.1 Both backends work
Newton (MJWarp rigid + VBD soft) and PhysX (FEM) both run `Isaac-Lift-Soft-Franka-v0`. Newton lifts the soft bar 0.049 → 0.567 m. PhysX benched OK at 1 env (53.6 steps/s) and 2048 envs (24.4 steps/s, 49.9k env-steps/s, 5.9 GB).

### 3.2 The tet mesh is non-deterministic — this dominates everything
The deformable has no pre-built tet mesh, so `isaaclab/sim/schemas/schemas.py:1263` tetrahedralises at spawn with `pytetwild.tetrahedralize` (fTetWild) — a **randomised, multithreaded algorithm called with no seed**. Six launches of the identical cuboid gave **61, 66, 67, 67, 70, 74 nodes**.

Consequence: the same command, same seed, same 5 N force, in two processes gave grasped widths of **62.06 mm vs 40.28 mm — a 35 % difference**. Within one process (mesh fixed) trials cluster to ~0.3 mm.

**Mitigation used:** monkeypatch `pytetwild.tetrahedralize` with a disk cache keyed on (surface, edge length) — see `grasp_sweep.py::_install_tet_cache`. Makes the mesh byte-identical across configs *and* processes.

**Proper fix (recommended, not yet done):** pre-tetrahedralise once and ship a `UsdGeom.TetMesh` under the deformable prim. `schemas.py:1160-1173` already prefers an existing TetMesh and only falls back to fTetWild when absent.

### 3.3 Mesh resolution matters more than the material
Fixed object, fixed 10 N, only `edge_length_fac` varied:

| nodes | 13 | 23 | 30 | 59 | 108 | 157 | 533 |
|---|---|---|---|---|---|---|---|
| grasped width (mm) | 45.9 | 19.1 | 53.9 | 53.5 | 58.4 | 59.0 | 30.9 |

**39.9 mm spread on a 50 mm object, non-monotonic, no convergence even at 533 nodes.** Larger than the effect of a 16× change in Young's modulus. Treat mesh resolution as a first-class parameter; never compare results across different meshes.

### 3.4 Hooke's law holds
0.05 m cube, fixed mesh, constant-force gripper:

| test | fitted slope | analytic `FL/EA` | R² |
|---|---|---|---|
| gap vs `1/E` | −8.96e5 mm·Pa | −2e5 | **0.9970** |
| gap vs `F` | −1.34 mm/N | −0.25 | **0.9861** |

Linear on both axes → the VBD material is Hookean here. But **~4.5–5.4× more compliant** than the analytic uniaxial prediction. The two ratios agreeing points at geometry, not a broken constitutive model: the Franka pads cover only ~1/5 of the 50×50 mm face, so the true load-bearing area is much smaller than `A = L²`. Calibrate against effective contact area, not nominal.

### 3.5 Finger-tip jitter: cause and fix
Symptom: finger gap oscillating **45 mm peak-to-peak**, never decaying.

Cause: to make the effort cap bind for constant-force control, finger stiffness was set to 2e4 → the drive is **permanently saturated**. A saturated actuator emits a constant force with its `d·q̇` damping term clipped away, so nothing removes energy. Against a springy fruit there is **no stable equilibrium** — it is a limit cycle.

Fix: command a fixed finger **position** (`--close_gap`) with effort headroom so the drive stays unsaturated → damped spring → settles.

| | tail p2p | jerk | tail std |
|---|---|---|---|
| force-saturated | 15.98 mm | 8.94 mm/step | 4.46 mm |
| position-hold | **0.118 mm** | **0.071 mm/step** | **0.034 mm** |

~130× improvement; fruit still deforms 2.25 mm at 6.6 N.

**Things that did NOT work** (recorded so they aren't retried):
- More joint damping → **diverges**. Damping is integrated explicitly: needs `d < 2m/dt ≈ 1.8` for a 15 g finger at 1/60 s. `d=200` sent the joint to −16 m.
- Armature 0.5 → traded fast chatter for a bigger slow oscillation (tail std 4.46 → 7.39).
- 240 Hz actuator rate → also worse (7.09). The 130 Hz contact-resonance aliasing argument was sound but not dominant.
- Raising contact damping `soft_contact_kd` 1e-5 → 1.0 → **solver blows up within a few steps**, arm Jacobian goes singular. Explicit contact damping is bounded by `c < 2m/dt`. Change only in small increments.

### 3.6 Asset root had to be pinned to 5.0
`apps/isaaclab.python.kit` lines ~309-311 changed from `Assets/Isaac/6.0` to `Assets/Isaac/5.0`.

Reason: NVIDIA renamed `Robots/FrankaEmika/panda_instanceable.usd` → `franka_panda.usda` in the 6.0 tree *after* beta2 shipped (only stale thumbnails of the old name remain). Every run died with `FileNotFoundError: USD file not found`. The 5.0 tree still has all three assets the task needs.

**Re-check this on Ubuntu** — see §7.4, because patch1 moves to Isaac Sim 6.0.1 and may have realigned the asset paths.

### 3.7 Strawberry deformable works
YCB `012_strawberry` scan → watertight → decimated 16384→1200 faces → USD `UsdGeom.Mesh` → spawned via `UsdFileCfg(deformable_props=..., physics_material=...)` → tetrahedralises to 207 nodes. At 10 N: y compresses 4.33 mm, z bulges 2.79 mm (correct Poisson response).

Run-to-run spread on the same config was **2.81 / 4.33 / 5.50 mm** — i.e. ±40 %. Average several runs before quoting any deformation figure.

---

## 4. Files to carry over

All small. **Do not copy** the venv (~15 GB), `videos/`, or `*.mp4`.

### In-repo (custom scripts, `scripts/environments/state_machine/`)
| File | Purpose |
|---|---|
| `lift_franka_soft_verify.py` | bounded self-check of the stock task |
| `lift_franka_soft_compare.py` | Newton vs PhysX + video (has the WGL patch) |
| `lift_franka_soft_physx_tune.py` | grasp-force / state-machine tuning |
| `bench_one.py` | single-point throughput benchmark |
| `grasp_determinism.py` | repeatability harness (constant EE pose, gripper only) |
| `grasp_sweep.py` | mesh / stiffness / force sweeps + **tet-mesh disk cache** |
| `grasp_strawberry.py` | strawberry deformable grasp + Newton render |

### Out-of-repo
| Path | Purpose |
|---|---|
| `assets/make_strawberry_usd.py` | YCB scan → watertight → decimate → `.usda` |
| `assets/strawberry_deformable.usda` | 46 KB, the generated asset |
| `assets/012_strawberry_google_16k.tgz` | 6.9 MB source scan (re-downloadable) |
| `sweeps/plot_sweeps.py`, `plot_strawberry.py`, `plot_jitter_fix.py`, `strawberry_frames.py`, `strawberry_zoom.py` | figures |
| `sweeps/*.csv`, `sweeps/*.npz` | raw sweep results (a few KB each) |
| `docs/` | this file + the earlier Newton-vs-PhysX doc |

### Local modifications to re-apply after a fresh clone
1. `apps/isaaclab.python.kit` — asset root `6.0` → `5.0` (3 lines), **only if §7.4 shows it's still needed**.
2. Nothing else. All other work is in new files, so a fresh clone + copying the scripts is clean.

---

## 5. Ubuntu setup, step by step

Assumes Ubuntu 22.04, an RTX GPU, and sudo.

### 5.1 Driver + CUDA toolkit
```bash
nvidia-smi                     # confirm driver; need a recent one for CUDA 12.8
sudo apt update && sudo apt install -y build-essential git git-lfs
git lfs install
```
Install the **CUDA 12.8 toolkit** (match torch's `cu128`) from NVIDIA's site, then:
```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version                 # must print 12.8
```
`nvcc` is required only by cuRobo — Isaac Sim itself does not need it.

### 5.2 Isaac Sim + Isaac Lab
Keep the path short and out of any synced folder (habit from Windows; harmless on Linux):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/isaac && cd ~/isaac
uv venv --python 3.12 --seed ~/isaac/venv
export VIRTUAL_ENV=~/isaac/venv
export OMNI_KIT_ACCEPT_EULA=YES

uv pip install "isaacsim[all,extscache]==6.0.1" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout v3.0.0-beta2.patch1
./isaaclab.sh --install
```
**Then re-pin torch** — the installer downgrades it to 2.10.0 and drops torchaudio:
```bash
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
```
Verify: `torch 2.11.0+cu128`, `cuda True`, `isaacsim 6.0.1`.

### 5.3 Smoke test
```bash
./isaaclab.sh -p scripts/environments/state_machine/lift_franka_soft_verify.py --num_envs 1 --max_steps 300
```
Expect `[VERIFY] SUCCESS ... object z: 0.049 -> ~0.5`.

On Ubuntu also check that **Newton CUDA-graph capture now succeeds** — grep the log for `libcudart`. If it's gone, re-run `bench_one.py` for Newton at 1/64/256/1024/2048 envs; the Windows numbers in the old doc should improve dramatically and the "PhysX wins by 65×" conclusion will likely reverse.

---

## 6. About "pushing to your repo" — read this

**Neither `C:\isaac_soft` nor `C:\isaac_soft\IsaacLab-3.0.0-beta2` is a git repository.** No `.git`, no remote, no history — it is an unpacked source tree, not a clone. So:

- There is **no repo of yours here to push to**, and nothing identifies it as `shuakang`'s fork.
- `gh` is not installed and no git identity (`user.name` / `user.email`) is configured on this machine.
- Consequently **nothing was pushed**, and I did not create a repo or push anywhere, since that is an outward-facing action needing an explicit target.

To set this up properly on Ubuntu:
```bash
# fork isaac-sim/IsaacLab on GitHub first, then:
git clone https://github.com/<you>/IsaacLab.git
cd IsaacLab
git remote add upstream https://github.com/isaac-sim/IsaacLab.git
git checkout -b <you>/soft-grasp v3.0.0-beta2.patch1
# copy the custom scripts from §4, then commit
```
Add a `.gitignore` for `videos/`, `*.mp4`, `*.npz`, `sweeps/*.png`, and any venv. Per the repo's own `AGENTS.md`: branch as `<username>/feature-desc`, never commit to `main`, never push to `origin` if that is upstream, and **no AI attribution lines in commit messages**.

If you want me to do this, give me the fork URL and confirm the account — I'll need `gh` installed and authenticated.

---

## 7. Syncing to the latest official IsaacLab

Current local: **v3.0.0-beta2**. Latest official: **v3.0.0-beta2.patch1** (2026-07-02) — one patch ahead.

### 7.1 What's in patch1
- Bumps **Isaac Sim to 6.0.1** (fixes + NuRec workflow improvements)
- `h5py >= 3.16.0`
- Cherry-picked Isaac Sim 6.0 streaming crash fix

### 7.2 How to sync
Do **not** try to git-merge the current folder — it has no history. On Ubuntu, clone fresh at `v3.0.0-beta2.patch1` (§5.2) and copy the seven custom scripts across. Our changes are additive except the one 3-line kit edit, so this is clean.

### 7.3 Isaac Sim version
Use `isaacsim==6.0.1` to match patch1, not `6.0.0.1`.

### 7.4 Re-check the asset root before re-applying the 5.0 pin
On the fresh clone, **first try stock 6.0 paths**:
```bash
python - <<'PY'
import urllib.request
base="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac"
for v in ("6.0","5.0"):
    for p in ("Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
              "Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda"):
        try:
            urllib.request.urlopen(f"{base}/{v}/{p}", timeout=20); print("OK  ", v, p)
        except Exception as e: print("MISS", v, p, type(e).__name__)
PY
```
If 6.0 now serves `panda_instanceable.usd`, **skip the 5.0 pin entirely** — it was a workaround for upstream churn, not something we want to keep.

---

## 8. cuRobo on Ubuntu

### 8.1 Install
```bash
sudo apt install -y git-lfs && git lfs install
git clone https://github.com/NVlabs/curobo.git
cd curobo
export VIRTUAL_ENV=~/isaac/venv
export TORCH_CUDA_ARCH_LIST="8.9"    # RTX 6000 Ada / RTX 4090 = 8.9; A100 = 8.0; H100 = 9.0
uv pip install tomli wheel ninja
uv pip install -e . --no-build-isolation      # ~20 min, builds CUDA kernels
python -m pytest .                            # verify
```

### 8.2 Ignore the docs' Isaac Sim path
cuRobo's documented Isaac Sim route assumes **Isaac Sim 4.0 + CUDA 11.8** and a `~/.local/share/ov/pkg/...` layout. That is stale against our 6.0.1 / CUDA 12.8 / torch 2.11 pip-based stack. Install into the same uv venv instead (§8.1) — Isaac Sim from pip *is* the Python environment, so there is no separate bundled interpreter to target.

### 8.3 The one real risk: Python 3.12
cuRobo is tested on **3.8–3.10** ("3.11+ may work but isn't tested"); Isaac Sim 6.0 forces **3.12**. The build is mostly torch C++/CUDA extensions, which usually don't care about the minor version, so it will probably work — but this is the step most likely to fail.

Fallbacks, in order of preference:
1. Patch `setup.py`/`pyproject.toml` `python_requires` if the only failure is a version guard.
2. **Planning service**: run cuRobo in a separate Python 3.10 venv exposing a tiny RPC (ZeroMQ/gRPC/stdin-JSON) that takes `(joint_state, target_pose, world_config)` and returns a trajectory. Isaac Lab calls it. Clean boundary, costs one process hop — trajectories are planned at ~1 Hz, so latency is irrelevant.
3. Docker: cuRobo ships Dockerfiles; run the planner in a container with the same RPC boundary.

---

## 9. Future work: the pick-and-place task

**Goal:** pick a fruit off the desk and place it in a pot, with cuRobo doing the motion planning, and collect demonstration data.

### 9.1 cuRobo integration (mirroring RoboTwin)
RoboTwin's `envs/robot/planner.py` is the reference. Its `CuroboPlanner` is engine-agnostic — it only touches joint states and poses, so it ports to Isaac Lab unchanged in substance:

```python
motion_gen_config = MotionGenConfig.load_from_robot_config(
    yml_path,                       # robot cfg: urdf, base/ee link, joint names
    world_config,                   # obstacles as cuboid/mesh dicts
    interpolation_dt=1/250,
    num_trajopt_seeds=1,
)
motion_gen = MotionGen(motion_gen_config); motion_gen.warmup()

start = JointState.from_position(torch.tensor(q).cuda().reshape(1,-1),
                                 joint_names=active_joints)
result = motion_gen.plan_single(start, CuroboPose.from_list([*p, *quat]),
                                MotionGenPlanConfig(max_attempts=10))
if result.success.item():
    traj = result.interpolated_plan.position.cpu().numpy()   # (T, n_joints)
```

Points to carry over:
- **Poses must be in the robot base frame.** RoboTwin transforms world→base and adds a `frame_bias` from the yml. In our env the IK action is already env-local (`root_pos_w - env_origins`) — do not double-apply.
- cuRobo ships a `franka.yml`; check its ee link/`frame_bias` against `panda_hand` + the task's `body_offset` of `[0, 0, 0.107]`.
- `plan_batch` exists for multi-goal planning — useful for grasp-pose candidates.
- Execute the trajectory by feeding waypoints to a **joint-position action**, not the differential-IK action currently in the task. The task's `ActionsCfg` uses `DifferentialInverseKinematicsActionCfg` (7-D pose + 1 gripper); planned joint trajectories want `JointPositionActionCfg` instead. This is the main env-side change.

### 9.2 World model for collision
cuRobo needs the obstacles. Minimum: table cuboid + pot. The pot is **concave** — do not hand cuRobo a convex box for it or the planner will refuse to enter the opening. Options: decompose the pot into 4 wall cuboids + 1 base cuboid (simplest, exact enough), or give cuRobo the mesh directly via its `mesh` world primitive.

### 9.3 The pot (self-generated)
Generate with trimesh, export as USD, spawn as a **static rigid** object:
```python
import trimesh, numpy as np
outer = trimesh.creation.cylinder(radius=0.070, height=0.090, sections=48)
inner = trimesh.creation.cylinder(radius=0.060, height=0.090, sections=48)
inner.apply_translation([0, 0, 0.010])        # leave a floor
pot = outer.difference(inner)                  # needs a boolean backend (manifold3d/blender)
```
Notes:
- `trimesh.boolean` needs a backend — `pip install manifold3d`.
- Opening (Ø 120 mm) must comfortably exceed the fruit (Ø ~45 mm) plus the finger pads.
- For physics collision use **convex decomposition** (`omni.convexdecomposition`) or an SDF collider; a single convex hull would seal the pot shut.
- Simplest robust alternative: skip booleans and assemble the pot from 1 base cylinder + N thin wall boxes arranged in a ring. Trivially convex-decomposable and exact for cuRobo's world model.

### 9.4 The fruit (MetaFood3D)
The Purdue site is **MetaFood3D** — 637 models, 108 categories, with nutrition labels. Download requires a **request form → approval → password**, so it cannot be automated. Someone must request access.

Until then use the YCB strawberry already built (`assets/strawberry_deformable.usda`). The pipeline in `make_strawberry_usd.py` is asset-agnostic: point it at any watertight mesh, it decimates → recentres → writes USD. Swapping in a MetaFood3D model is a one-line path change.

Decide early: **rigid or deformable fruit?**
- Rigid is far simpler and is what a pick-and-place data-collection task normally wants.
- Deformable reuses everything in §3 but is slower, and the grasp must use the **position-hold** mode (§3.5) or the fingers will chatter.

### 9.5 Data collection
RoboTwin's pattern: script the task, plan each segment with cuRobo, execute, record. Per episode log: joint positions/velocities, ee pose, gripper state, object pose, camera frames if needed, plus a success flag. Success = fruit's final position inside the pot's rim radius and below the rim height. Note the ±40 % run-to-run variance (§3.2, §3.7) — with a deformable fruit, expect a non-trivial failure rate and record it rather than tuning until it looks perfect.

### 9.6 Suggested order
1. Ubuntu env up, smoke test passes (§5).
2. Re-check asset root (§7.4).
3. Re-run Newton benchmarks — expect the Windows CUDA-graph penalty to vanish (§1).
4. Install cuRobo, run its own `motion_gen_reacher` example (§8).
5. Swap the task's action term to joint-position control.
6. Wrap `CuroboPlanner` for Isaac Lab; plan a free-space reach, no objects.
7. Add the pot + table to the cuRobo world; verify planned paths avoid them.
8. Rigid fruit pick-and-place end to end.
9. Only then (optional) switch the fruit to deformable.
10. Data collection loop + success metric.

---

## 10. Gotchas checklist

- [ ] Re-pin torch **after** `isaaclab.sh --install` — it silently downgrades to 2.10.0 and removes torchaudio.
- [ ] Set `OMNI_KIT_ACCEPT_EULA=YES` or Kit hangs.
- [ ] Never compare deformable results across different tet meshes (§3.2). Use the disk cache or a pre-built TetMesh.
- [ ] Don't raise contact damping (`soft_contact_kd`) beyond ~1e-4 without testing — it diverges (§3.5).
- [ ] Don't raise joint damping on the fingers — explicit integration bounds it at `d < 2m/dt ≈ 1.8` (§3.5).
- [ ] For any deformable grasp use **position-hold**, not force-saturation, unless you specifically need constant force (§3.5).
- [ ] The env's gravity is **off** by default; `--gravity` in the tune script enables it. With gravity on, the arm's default PD (~80 stiffness) sags and never reaches the object — raise arm stiffness to ~800.
- [ ] The Franka arm's default stiffness is too soft to hit a small object; a 0.3 m bar forgives lateral IK error, a 45 mm fruit does not.
- [ ] `ViewerCfg.origin_type="asset_root"` is ignored headless — `eye`/`lookat` are absolute world coordinates there.
- [ ] Newton's GL viewer ignores USD `PreviewSurface` materials; objects render in default colours. Use `--viz kit` for correct materials.
- [ ] Episode length is 5 s (300 steps at 60 Hz); long trials need `episode_length_s` raised or they reset mid-run.

---

## 11. Open questions for whoever picks this up

1. **GitHub target** — which account/repo should this be pushed to? Nothing is a git repo today (§6).
2. **MetaFood3D access** — has the request been filed? Which fruit category?
3. **Rigid or deformable fruit** for the pick-and-place (§9.4)?
4. **Which robot** — stay with the Franka Panda, or match RoboTwin's Aloha-Agilex dual-arm?
5. **How much data** — episode count and which modalities (state only, or RGB-D too)?
