# Isaac Lab Offline Training Setup
> Complete guide for training Isaac Lab environments offline using locally downloaded assets.

## 🎯 Overview
#### The offline training system enables you to train Isaac Lab environments without internet connectivity by using locally downloaded assets. This system:
- ✅ Works with any robot - No hardcoded paths needed
- ✅ Single flag - Just add --offline to your training command
- ✅ Automatic fallback - Uses Nucleus if local asset is missing
- ✅ Maintains structure - Mirrors Nucleus directory organization locally

## 📦 Requirements
- Isaac Lab installed and working
- Isaac Sim 5.0 or later
- 2-20 GB free disk space (depending on assets downloaded)
- Internet connection for initial asset download

## 🚀 Quick Start
### 1. Download essential assets (one-time, `all` ~30 GB)
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
### 2. Train completely offline with any robot via the `--offline` flag (also works with `/play`)
#### Supported for: `rl_games`, `rsl_rl`, `sb3`, `skrl`, and `sim2transfer`
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 128 \
    --offline

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 128 \
    --checkpoint logs/rsl_rl/<robot>_flat/<timestamp>/model_<num>.pt \
    --video \
    --video_length 1000 \
    --offline
```
### 3. Run various demos and tutorials with `--offline` flag

```
./isaaclab.sh -p scripts/tutorials/01_assets/run_deformable_object.py --offline
```

#### _Note: For offline training, assets that cannot be found in `offline_assets` will attempted to be fetched from the [Nucleus Server](https://docs.omniverse.nvidia.com/nucleus/latest/index.html)._

## 📁 Asset Layout
#### Offline assets are organized to mirror Nucleus (`ISAAC_NUCLEUS_DIR` & `ISAACLAB_NUCLEUS_DIR`) under the `offline_assets` directory, meaning that no code changes are required for offline running! We flatten `Isaac/IsaacLab/` to just the category names (`Robots/`, `Controllers/`, etc.) for cleaner local structure. This happens in `asset_resolver.py`, where the resolver maintains a 1:1 mapping between Nucleus and local storage.

```
IsaacLab/
├── source/isaaclab/isaaclab/utils/
│   └── asset_resolver.py           # Core resolver
├── scripts/setup/
│   └── download_assets.py          # Asset downloader
└── offline_assets/
    ├── ActuatorNets/
    ├── Controllers/
    ├── Environments/               # Ground planes
    │   └── Grid/
    │       └── default_environment.usd
    ├── Materials/                  # Textures and HDRs
    │   └── Textures/
    │       └── Skies/
    ├── Mimic/
    ├── Policies/
    ├── Props/                      # Markers and objects
    │   └── UIElements/
    │       └── arrow_x.usd
    └── Robots/                     # Robot USD files
        ├── BostonDynamics/
        │   └── spot/
        │       └── spot.usd
        └── Unitree/
            ├── Go2/
            │   └── go2.usd
            └── H1/
                └── h1.usd
```
