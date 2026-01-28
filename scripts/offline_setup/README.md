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









### Proposal

Add offline training support to Isaac Lab that enables training environments without internet connectivity. The feature should automatically redirect asset paths from Nucleus/S3 servers to local storage while maintaining the same directory structure, requiring zero configuration changes to existing environments.

Core capabilities:
- One-time asset download script that mirrors Nucleus directory structure locally
- Automatic path resolution from Nucleus URLs to local filesystem
- Single `--offline` flag for all training scripts
- Graceful fallback to Nucleus if local asset is missing
- Works with any robot and environment without hardcoded paths

### Motivation

**Current Problem:**
Training Isaac Lab environments requires constant internet connectivity to load assets from Nucleus/S3 servers. This creates several critical issues:

1. **Airgapped Environments**: Impossible to train in secure/classified facilities that prohibit internet access
2. **Network Reliability**: Training fails or becomes extremely slow on unstable networks
3. **Reproducibility**: Repeated downloads slow iteration and depend on external server availability
4. **Development Workflow**: Researchers waste time waiting for assets during rapid prototyping
5. **DGX Spark is Rooted**: I should be able to take my Spark anywhere regardless of internet connectivity and live out Isaac Lab and Isaac Sim to it's fullest

**User Story:**
"I'm always frustrated when I need to train in an airgapped lab environment or on an unstable network connection. I have to create custom configs with hardcoded local paths for each robot, which results in unmaintainable code duplication and can break when I switch robots or update Isaac Lab. I would like to take Isaac Lab on the go and quickly demo anything in an independent localized ecosystem without nit-picking configurations and dealing with asset management."

**Core User Needs:**
- Train without internet connectivity
- Avoid hardcoded paths in environment configs
- Work seamlessly with any robot
- Maintain same configs for both online and offline modes

### Alternatives

**Alternative 1: Hardcoded Local Paths** (Current workaround)
```python
# Separate config file per robot
UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
```
**Problems:**
- ❌ Requires separate config for each robot
- ❌ ~30 lines of boilerplate per robot
- ❌ Breaks when Isaac Lab updates
- ❌ Not maintainable

**Alternative 2: Environment Variables**
```bash
export ISAAC_ASSETS_DIR="/local/path"
```
**Problems:**
- ❌ Doesn't work with S3 URLs
- ❌ Requires Isaac Sim core changes
- ❌ Can't have fallback to Nucleus
- ❌ Not transparent to users

**Alternative 3: Manual Asset Copying**
Copy assets manually and update configs.
**Problems:**
- ❌ Error-prone manual process
- ❌ No directory structure guidelines
- ❌ Still requires config changes

**Proposed Solution Benefits:**
- ✅ Zero code or config changes
- ✅ Works with all robots and environments automatically
- ✅ Single flag: `--offline` for all training environments (`rl_games`, `rsl_rl`, `sb3`, `skrl`, and `sim2transfer`)
- ✅ Automatic fallback
- ✅ Easy `offline_asset` plug and play following Nucleus structure
- ✅ 90% code reduction

### Build Info

- Isaac Lab Version: main branch (as of January 2026)
- Isaac Sim Version: 5.1.0

### Additional Context

**Use Cases:**
1. **Defense/Aerospace**: Training in classified airgapped facilities
2. **Remote Locations**: Field robotics research with limited connectivity
3. **Development**: Rapid iteration without network delays and firewall interruptions
4. **CI/CD**: Reproducible builds without external dependencies
5. **Workshops/Tutorials**: Teaching without relying on conference WiFi (i.e. localized everything on DGX Spark)

**Technical Approach:**
The implementation uses a dual-layer strategy:
- **Monkey patching**: Intercepts asset loads at spawn config instantiation (90% coverage)
- **Config patching**: Explicitly modifies pre-loaded configs (10% coverage)

This ensures ~100% asset coverage without modifying environment configs.

**Expected Directory Structure:**
```
IsaacLab/
├── source/isaaclab/isaaclab/utils/
│   └── asset_resolver.py           # Core resolver
├── scripts/setup/
│   └── download_assets.py          # Asset downloader
└── offline_assets/
    ├── ActuatorNets/...
    ├── Controllers/...
    ├── Environments/               # Ground planes
    │   └── Grid/
    │       └── default_environment.usd
    ├── Materials/                  # Textures and HDRs
    │   └── Textures/
    │       └── Skies/
    ├── Mimic/...
    ├── Plocies/...
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

Dynamically pulls and mirrors Nucleus structure for seamless path resolution.

### Checklist

- [x] I have checked that there is no similar issue in the repo (**required**)

### Acceptance Criteria

- [x] Asset download script that mirrors Nucleus directory structure to local storage (offline_assets)
- [x] Automatic path resolver that redirects Nucleus URLs to local filesystem
- [x] Optional `--offline` flag added to all training scripts (RSL-RL, SB3, SKRL, RL Games)
- [x] Monkey patching of Isaac Lab spawn configs (UsdFileCfg, GroundPlaneCfg, PreviewSurfaceCfg)
- [x] Config patching for pre-loaded environment configs
- [x] Graceful fallback to Nucleus for missing assets
- [x] Support for versioned Nucleus URLs (e.g., `/Assets/Isaac/5.1/...`)
- [x] Documentation including setup guide, usage examples, and troubleshooting
- [x] Works with any robot without hardcoded paths
- [x] Zero breaking changes - existing code continues to work
- [x] Manual testing completed across multiple robots (Go2, H1, ANYmal)
- [x] Verification in complete offline mode (no internet connectivity)

**Definition of Done:**
A user can download all, or select assets by running `./isaaclab.sh -p scripts/offline_setup/download_assets.py --categories all`, then train any robot completely offline by simply adding `--offline` to their training command, with no code or config changes required.
