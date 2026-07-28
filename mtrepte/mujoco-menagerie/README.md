# MuJoCo Menagerie Benchmark Branch

This branch ports the MuJoCo Menagerie asset swap onto `mtrepte/mujoco_menagerie_v2`, based on current IsaacLab `develop`.

The goal is to benchmark IsaacLab training environments with robot USD assets from:

```text
${ISAAC_NUCLEUS_DIR}/Samples/Mujoco_Menagerie
```

The branch intentionally does not fall back to legacy IsaacLab robot USDs if a Menagerie path or physics variant is missing. Broken paths should fail during launch so they are visible in local smoke runs, Docker runs, and CI-style benchmarks.

## What Changed

- Core robot asset configs now point to MuJoCo Menagerie USD/USDC assets for the covered robots.
- Training launchers accept `--menagerie-physics-variant` to select Menagerie USD `Physics` variants.
- The benchmark harness has a Menagerie-specific config at `source/isaaclab_tasks/test/benchmarking/mujoco_configs.yaml`.
- The benchmark test can emit KPI payloads via `--save_kpi_payload`.

## Primary Smoke Command

Run from the repo root:

```bash
./isaaclab.sh -p -m pytest -s -x \
  source/isaaclab_tasks/test/benchmarking/test_environments_training.py \
  --config_path mujoco_configs.yaml \
  --mode test \
  --workflows rsl_rl \
  --sim-backend physx \
  --save_kpi_payload
```

For Newton/MuJoCo physics variant coverage, switch to:

```bash
--sim-backend newton
```

For longer KPI collection, switch to:

```bash
--mode benchmark
```

## Docker A/B Benchmark Shape

Use the same `mujoco_configs.yaml` in both images:

- Base image: current `develop`, original IsaacLab robot assets.
- Menagerie image: this branch, Menagerie robot assets and explicit `--sim-backend physx` or `--sim-backend newton`.

Compare generated KPI payloads from `logs/kpi.json`.

## Docs

- `asset_mapping.txt`: IsaacLab robot config to Menagerie asset path mapping.
- `asset_tasks.txt`: representative tasks that exercise each Menagerie-backed asset.
- `training_benchmark_catalog.txt`: full benchmark matrix and command notes.
