# Isaac Lab Offline Training Setup
> Complete guide for training Isaac Lab environments offline using locally downloaded assets.

## 🎯 Overview
#### The offline training system enables you to train Isaac Lab environments without internet connectivity by using locally downloaded assets. This system:
- ✅ Works with any robot - No hardcoded paths needed
- ✅ Single flag - Add `--offline` to your training, tutorial, and demo commands
- ✅ Fallback flag - Add `--offline-permissive` to your commands to fallback to Nucleus for asset not found in `offline_assets`
- ✅ Maintains structure - Mirrors Nucleus directory organization locally

## 📦 Requirements
- Isaac Lab installed and working
- Isaac Sim 5.0 or later
- 2-60 GB free disk space (depending on assets downloaded)
- Internet connection for initial asset download

## 🚀 Quick Start
### 1. Download essential assets (one-time, `all` ~60 GB)
#### Assets download to the `~/IsaacLab/offline_assets` directory: `cd ~/IsaacLab`
```
./isaaclab.sh -p scripts/offline_setup/download_assets.py \
    --categories all
```
#### _Alternative Note: Category fields can be specified separately_
```
./isaaclab.sh -p scripts/offline_setup/download_assets.py \
    --categories Robots --subset Unitree
```

### Successful Downloads
```
======================================================================
📊 Downloaded Assets Summary
======================================================================
Location: ~/IsaacLab/offline_assets

✓ Environments           34,437 files       17.0 GB
✓ IsaacLab                4,197 files        9.3 GB
✓ Materials               1,918 files      537.0 MB
✓ People                  3,085 files        9.4 GB
✓ Props                   2,507 files        4.2 GB
✓ Robots                  4,687 files        5.3 GB
✓ Samples                 6,601 files       10.4 GB
✓ Sensors                   772 files      256.7 MB
======================================================================
TOTAL                  58,204 files       56.3 GB
======================================================================

✅ Complete! Use --offline flag with Isaac Lab commands.

Offline mirror: ~/IsaacLab/offline_assets
```

### 2. Train completely offline with any robot via the `--offline` flag (also works with `/play`)
#### Supported for: `rl_games`, `rsl_rl`, `sb3`, `skrl`, and `sim2transfer`
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --num_envs 64 \
    --max_iterations 10 \
    --offline

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Velocity-Flat-H1-v0 \
    --num_envs 64 \
    --max_iterations 10 \
    --offline

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 128 \
    --checkpoint logs/rsl_rl/<robot>_flat/<timestamp>/model_<num>.pt \
    --video \
    --video_length 1000 \
    --offline
```
#### Run various demos and tutorials with `--offline` flag

```
./isaaclab.sh -p scripts/demos/quadrupeds.py --offline

./isaaclab.sh -p scripts/demos/arms.py --offline

./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py --offline
```

#### Strict mode (default) - fails immediately if asset not found locally
```
./isaaclab.sh -p train.py --task Go2 --offline
```

#### Permissive mode - warns and falls back to Nucleus if asset not found
```
./isaaclab.sh -p train.py --task Go2 --offline-permissive
```

#### Environment variables work too
```
OFFLINE=1 ./isaaclab.sh -p train.py --task Go2
OFFLINE_PERMISSIVE=1 ./isaaclab.sh -p train.py --task Go2
```

### Missing Assets
```
======================================================================
[OfflineAssetResolver] ✗ ASSET NOT FOUND (offline mode)
======================================================================
Missing:  IsaacLab/Robots/Unitree/Go2/go2.usd
Expected: ~/IsaacLab/offline_assets/IsaacLab/Robots/Unitree/Go2/go2.usd

To download this asset, run:
  ./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories IsaacLab/Robots
```

#### _Note: For offline training, assets that cannot be found in `offline_assets` will attempted to be fetched from the [Nucleus Server](https://docs.omniverse.nvidia.com/nucleus/latest/index.html)_ when using the `--offline-permissive` flag.

## 📁 Asset Layout
#### Offline assets are organized to mirror Nucleus (`ISAAC_NUCLEUS_DIR` & `ISAACLAB_NUCLEUS_DIR`) under the `offline_assets` directory, meaning that no code changes are required for offline running! We flatten `Isaac/IsaacLab/` to just the category names (`Robots/`, `Controllers/`, etc.) for cleaner local structure. This happens in `asset_resolver.py`, where the resolver maintains a 1:1 mapping between Nucleus and local storage.

```
IsaacLab/
├── source/isaaclab/isaaclab/utils/
│   └── asset_resolver.py           # Core resolver
├── scripts/setup/
│   └── download_assets.py          # Asset downloader
└── offline_assets/                 ← Mirror of Isaac/
├── IsaacLab/
│   ├── Robots/
│   │   ├── ANYbotics/
│   │   │   ├── ANYmal-B/
│   │   │   ├── ANYmal-C/
│   │   │   └── ANYmal-D/
│   │   ├── Unitree/
│   │   └── FrankaEmika/
│   ├── ActuatorNets/
│   ├── Controllers/
│   └── Policies/
├── Props/
│   └── UIElements/
├── Environments/
│   └── Grid/
├── Materials/
│   └── Textures/
└── Robots/                        ← Isaac Sim robots (different from IsaacLab/Robots)
```
