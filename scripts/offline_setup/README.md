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
### 1. Download essential assets (one-time, ~2-4 GB)
#### Assets download to the `~/IsaacLab/offline_assets` directory: `cd ~/IsaacLab`
```
./isaaclab.sh -p scripts/offline_setup/download_assets.py \
    --categories all
```
#### _Optional Note: Specific category fields can be specified separately_
```
./isaaclab.sh -p scripts/offline_setup/download_assets.py \
    --categories Props Robots Environments Materials Controllers ActuatorNets Policies Mimic
```
### 2. Train completely offline with any robot via the `--offline` flag (also works with `/play`)
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 128 \
    --offline

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/unitree_go2_flat/2026-01-27_14-58-33/model_800.pt \
    --video \
    --video_length 1000
    --offline
```
#### _Note: For offline training, assets that cannot be found in `offline_assets` will be fetched from the [Nucleus Server](https://docs.omniverse.nvidia.com/nucleus/latest/index.html)._

## 📁 Asset Layout
#### Offline assets are organized to mirror Nucleus (`ISAAC_NUCLEUS_DIR` & `ISAACLAB_NUCLEUS_DIR`) meaning that no code changes are required!

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
    ├── Plocies/
    ├── Props/                      # Markers and objects
    │   └── UIElements/
    │       └── arrow_x.usd
    └── Robots/                     # Robot USD files
        ├── Unitree/
        │   ├── Go2/
        │   │   └── go2.usd
        │   └── H1/
        │       └── h1.usd
        └── ANYbotics/
            └── ANYmal-D/
                └── anymal_d.usd
```
