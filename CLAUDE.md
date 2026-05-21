# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is the **Newton-physics experimental branch** of Isaac Lab (NVIDIA's GPU robotics-RL framework), built on Isaac Sim 5.0. The classic Isaac Lab feature set (manager-based + direct RL envs, assets, RL framework wrappers) lives alongside the in-tree Newton integration. This branch is under active development and not feature-complete relative to `main`; many classic envs are not yet ported.

On top of stock Isaac Lab, this checkout also carries an in-development **catheter fluoroscopy simulator** (`isaaclab_newton` extension) — an XPBD Cosserat-rod solver coupled to a Slang GPU DRR renderer (`fluorosim`, sibling repo `i4h-sensor-simulation-internal/fluoro-simulator`). Expect heavy churn in `source/isaaclab_newton/` and `source/isaaclab_newton/docs/`.

## The `isaaclab.sh` wrapper

Almost everything goes through `./isaaclab.sh` (alias `isaaclab` once the conda env is activated). It locates the right Python (conda > uv venv > Isaac Sim kit Python > system) via `_isaac_sim/` symlink or pip-installed `isaacsim-rl`, manages an `LD_PRELOAD` shim for the Torch-bundled libgomp, and forwards args to the right tool.

```bash
./isaaclab.sh -c [env_name]    # create conda env (default: env_isaaclab) from environment.yml
./isaaclab.sh -u [env_name]    # create uv venv instead of conda
./isaaclab.sh -i [framework]   # install all source/ extensions editable + RL framework deps
                               # (default "all"; or rsl_rl, rl_games, sb3, skrl, none)
./isaaclab.sh -p script.py …   # run Python with the right interpreter — use this, NOT bare `python`
./isaaclab.sh -s …             # launch isaac-sim.sh with source/ as the extension folder
./isaaclab.sh -t [pytest-args] # run pytest tools/ (this is the project-wide test target)
./isaaclab.sh -f               # run pre-commit on all files (format + lint)
./isaaclab.sh -d               # build sphinx docs into docs/_build/current/
./isaaclab.sh -v               # regenerate .vscode/ settings from template
./isaaclab.sh -n               # template generator (new external project / internal task)
./isaaclab.sh -o …             # forwards to docker/container.sh
```

Two non-obvious behaviors:

- `-i` runs `find -L source -mindepth 1 -maxdepth 1 -type d` and pip-installs each `setup.py` it finds editable. Adding a new `source/<extension>/` directory with a `setup.py` is enough to wire it in.
- The script pins `torch==2.7.0+cu128` (x86) / `2.9.0+cu130` (ARM) and uninstalls any Isaac-Sim-bundled `numpy<2`. Don't fight this — let `-i` reinstall torch/numpy if you see import errors after upgrades.

## Source layout — what each extension is for

`source/` holds editable Python packages. Each has its own `setup.py`, `config/extension.toml`, `docs/`, and `test/`. The `-i` flag installs all of them.

- `isaaclab/` — core framework: scene, sim, assets, sensors, managers, terrains, controllers, RTX renderer wrapper, USD utilities. Everything else depends on this.
- `isaaclab_assets/` — robot/sensor configs (USD paths + dataclass cfgs). Pure config, no logic.
- `isaaclab_rl/` — adapters to RSL-RL, SKRL, RL-Games, Stable-Baselines3. The `[framework]` extras in `setup.py` are what `isaaclab.sh -i <framework>` selects.
- `isaaclab_tasks/` — concrete environments. Two flavors live side-by-side:
  - `manager_based/` — env composed declaratively from MDP managers (observation/reward/termination/…); the recommended pattern.
  - `direct/` — env subclasses `DirectRLEnv` and implements `_get_observations` / `_get_rewards` / etc. itself; better for tightly-coupled tasks.
  - Each task registers via `gym.register(...)` in its `__init__.py`; `isaaclab_tasks/__init__.py` walks the tree and imports them at load time (skipping `_BLACKLIST_PKGS`).
- `isaaclab_tasks_experimental/` — same shape, lower stability bar.
- `isaaclab_experimental/` — experimental core features.
- `isaaclab_newton/` — Newton physics integration **and** the catheter / XCath work. See below.

`scripts/` holds runnable drivers (not packages):
- `scripts/reinforcement_learning/{rsl_rl,skrl,rl_games,sb3}/{train,play}.py` — standard RL drivers; all take `--task <gym-id> --num_envs <N>` plus AppLauncher args (`--headless`, `--enable_cameras`, …). They use `hydra_task_config` to load the env+agent config registered by the task.
- `scripts/environments/{list_envs,random_agent,zero_agent}.py` — smoke-test drivers.
- `scripts/benchmarks/` — perf scripts and shell wrappers for the Newton-alpha eval matrix.

## The `isaaclab_newton` extension

Three things share this extension:

1. **Newton physics backend wrappers**: `assets/`, `actuators/`, `sensors/`, `envs/` mirror the `isaaclab/` shapes but route into Newton/MuJoCo-warp instead of PhysX. `isaaclab.Backend` enum (`NEWTON | PHYSX`) is the dispatch switch.
2. **Rod solver stack** (`solvers/`) — independent of Isaac Sim, pure Warp/Torch:
   - `rod_solver.py` / `rod_data.py` / `rod_kernels.py` — direct position-based stiff-rod solver (Deul et al. 2018).
   - `xpbd_rod_solver.py` — XPBD Cosserat rods; captures the substep loop into a CUDA graph on first `step()` call. `reset_cuda_graph()` invalidates the capture (the Reset button does this after rewriting initial positions).
   - `xcath_rod_solver.py` — XPBD + vessel mesh collision (SDF + AABB/mesh-edge paths) + track-guided insertion for catheter-in-vessel simulation.
   - `newton_xpbd_rod_wrapper.py` — adapter to upstream Newton's `SolverXPBDRod`. Requires the `xpbd_rod` extra (`pip install -e .[xpbd_rod]`), which pins Newton to PR #1981.
3. **Interactive fluoroscopy demo** (`examples/interactive_catheter_fluoro.py`) — Gradio web UI at `localhost:7860`. After `pip install -e source/isaaclab_newton`, a `xcath-fluoro` console script is registered. The demo also depends on the sibling `fluorosim` package (`pip install -e ../i4h-sensor-simulation-internal/fluoro-simulator[all]`); if not installed, the script tries to locate it relative to its own path. See `source/isaaclab_newton/docs/INTERACTIVE_FLUORO_README.md` for the full architecture writeup (5-step unified sim loop, DSA pipeline, coordinate transforms).

`source/isaaclab_newton/examples/` is a grab-bag of demo / smoke-test scripts (rod visualizers, training entry points like `train_catheter_state.py`). They're meant to be invoked via `./isaaclab.sh -p source/isaaclab_newton/examples/<name>.py`.

## Tests

```bash
./isaaclab.sh -t                                           # all tests under tools/
./isaaclab.sh -p -m pytest source/isaaclab_newton/test     # newton extension tests only
./isaaclab.sh -p -m pytest path/to/test_file.py::TestClass::test_name  # single test
./isaaclab.sh -p -m pytest -m isaacsim_ci                  # tests gated on Isaac Sim CI marker
```

`pytest.ini` defines the `isaacsim_ci` marker — use it for tests that require a live Isaac Sim. Each extension's `test/` folder may have its own `conftest.py` that adjusts `sys.path` for local helper modules (e.g. `source/isaaclab_newton/test/conftest.py` adds the articulation-data helpers).

## Formatting / lint

`./isaaclab.sh -f` runs the full pre-commit stack (config: `.pre-commit-config.yaml`):

- **black** with `--line-length 120 --unstable`
- **isort** with `--profile black`; the section order is configured in `pyproject.toml` (`STDLIB → THIRDPARTY → ASSETS_FIRSTPARTY → FIRSTPARTY → EXTRA_FIRSTPARTY → TASK_FIRSTPARTY → LOCALFOLDER`). `numpy`, `torch`, `gymnasium`, `warp`, `pxr`, etc. are pre-classified — don't hand-order imports between these groups.
- **flake8** (+ `flake8-simplify`, `flake8-return`)
- **pyupgrade** `--py310-plus` (with a hard-coded exclude list for files that break Torch's union-type aliasing)
- **codespell** with project word allowlist in `pyproject.toml`
- **insert-license** — every `.py` / `.yaml` must start with the BSD-3 header from `.github/LICENSE_HEADER.txt`. Files under `source/isaaclab_mimic/` use the Apache-2.0 header instead.

Pyright runs in `basic` mode targeting Python 3.11; `reportMissingImports`, `reportMissingModuleSource`, and `reportGeneralTypeIssues` are all disabled (CI doesn't have the deps installed and dataclass `MISSING` sentinels trip the general-issues check).

## Environment / version gotchas

- Python is **3.11** for Isaac Sim ≥ 5.0; the `isaaclab.sh -c` flow auto-patches `environment.yml` down to 3.10 if it detects Isaac Sim 4.5.
- Isaac Sim is located via `_isaac_sim/` symlink (binary install) or the `isaacsim-rl` pip package. If both are missing, `-p`, `-s`, `-i` will all error out.
- This branch's `_isaac_sim` must be 5.0+. See the table in `README.md` for the Isaac Lab ↔ Isaac Sim compatibility matrix.
- ARM (aarch64) builds get a separate torch pin and a temporary `LD_PRELOAD` unset during `-i` to dodge install-time conflicts (see `begin_arm_install_sandbox` in `isaaclab.sh`).

## Conventions worth knowing

- **Never invoke `python script.py` directly** for anything that imports `isaaclab*` or Isaac Sim; always go through `./isaaclab.sh -p script.py`. The wrapper sets `RESOURCE_NAME`, `LD_PRELOAD`, and conda activation hooks that the kit Python depends on.
- **Adding a new task**: create `source/isaaclab_tasks/isaaclab_tasks/{manager_based,direct}/<your_task>/`, add `__init__.py` with `gym.register(...)`, and register the agent config entry points (`rsl_rl_cfg_entry_point`, `skrl_cfg_entry_point`, …) in the same file. `isaaclab_tasks/__init__.py`'s auto-import will pick it up.
- **Adding a new extension**: create `source/<name>/setup.py` + `config/extension.toml`; `./isaaclab.sh -i` will install it editable on the next run.
- **The `unstable` flag is intentional** in the black config — don't strip it.
