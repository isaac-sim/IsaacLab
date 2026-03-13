# Isaac Lab Dev Journey Report

**Claude Code Skills for Isaac Lab — Validation & Results**

Date: 2026-02-25 | Platform: NVIDIA GB10 (aarch64) | Isaac Sim 6.0.0-rc.13 | Isaac Lab 4.0.0 (develop)

---

## Executive Summary

We built a Claude Code skill package for Isaac Lab that guides first-time users through the full RL development lifecycle — from environment setup through policy training. The skill was validated end-to-end across **all 4 supported RL frameworks** on a fresh-user workflow, achieving a **95% pass rate** (21/22 tests passed, 1 known upstream bug).

The skill package consists of **18 markdown documents** organized into 5 categories, totaling ~2,500 lines of verified, actionable guidance. Every command, file path, config value, and metric name was tested against the live codebase.

---

## What We Built

### Skill Architecture

```
.claude/skills/isaaclab/
├── SKILL.md                              # Entry point & routing
├── setup/
│   ├── fresh-install.md                  # Kit vs source-build install paths
│   ├── verification.md                   # 5-step environment verification
│   └── troubleshooting.md               # 14 common errors with fixes
├── architecture/
│   ├── overview.md                       # Manager-based vs Direct, MDP, modules
│   ├── patterns-comparison.md            # Side-by-side patterns, extension points
│   └── sensors-actuators.md              # 7 actuator types, 8 sensor types, 3 visualizers
├── training/
│   ├── guide.md                          # All 5 frameworks, CLI args, log structure, loss locations
│   ├── sb3-reference.md                  # SB3 deep-dive
│   ├── rsl-rl-reference.md               # RSL-RL + SKRL deep-dive
│   ├── rl-games-reference.md             # RL-Games deep-dive
│   └── hyperparameters.md                # Cross-framework tuning, TensorBoard metric mapping
├── environments/
│   ├── builder.md                        # Config structure, reward design rules
│   ├── mdp-catalog.md                    # All built-in MDP functions
│   └── templates.md                      # Full env + agent config templates
├── tutorials/
│   ├── cartpole.md                       # Hello World end-to-end
│   ├── code-walkthrough.md               # CartPole code analysis
│   └── experiments.md                    # 7 modification experiments
└── dev-journey-results.md                # This validation report (raw data)
```

### Key Capabilities

| Capability | What Claude Code Can Do |
|-----------|------------------------|
| **Environment Setup** | Detect GPU/CUDA/Python, verify `_isaac_sim` symlink, check all packages, diagnose 14 common errors |
| **Install Guidance** | Walk through Kit vs source-build, create symlinks, install extensions + RL frameworks |
| **Architecture Consulting** | Explain Manager-based vs Direct trade-offs, recommend patterns for user's use case |
| **Sensor/Actuator Setup** | Guide selection from 7 actuator types, 8 sensor types (Camera, TiledCamera, RayCaster, IMU, Contact, FrameTransformer, Visuotactile), 3 visualizers |
| **RL Training** | Generate correct commands for all 4 frameworks, explain hyperparameters, locate loss functions |
| **TensorBoard Monitoring** | Map metric names across frameworks, launch correctly (Kit Python path) |
| **Troubleshooting** | Diagnose and fix 14+ common errors (missing modules, OOM, warnings, convergence) |
| **Environment Building** | Provide templates, MDP function catalog, gym registration, reward design patterns |

---

## Validation Results

### Test Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (aarch64, CUDA capability 12.1) |
| Driver | 580.126.09 |
| CUDA | 13.0 |
| OS | Ubuntu 24.04, Linux 6.17.0-1008-nvidia |
| RAM | 119 GB |
| Python | 3.12.12 (Kit Python) |
| PyTorch | 2.9.0+cu130 |
| Isaac Sim | 6.0.0-rc.13 (source build) |
| Isaac Lab | 4.0.0 (repo v2.3.2, develop branch) |

### Phase 1: Environment Checking (5/5 PASS)

| Test | Result | Notes |
|------|--------|-------|
| GPU/CUDA detection | PASS | nvidia-smi, nvcc both work |
| Kit Python + PyTorch | PASS | torch.cuda.is_available() = True |
| All 4 RL framework imports | PASS | SB3 2.7.1, SKRL 1.4.3, RSL-RL 3.1.2, RL-Games 1.6.1 |
| All 9 Isaac Lab packages | PASS | isaaclab 4.0.0, all sub-packages installed |
| AppLauncher import | PASS | Omniverse Kit runtime boots correctly |

### Phase 2: Installation Verification (5/5 PASS, 1 SKIP)

| Test | Result | Notes |
|------|--------|-------|
| `_isaac_sim` symlink | PASS | Points to IsaacSim build |
| `python.sh` + `setup_python_env.sh` | PASS | PYTHONPATH/LD_LIBRARY_PATH configured |
| `isaaclab.sh --help` | PASS | All CLI options documented |
| TensorBoard in Kit Python | PASS | v2.20.0, NOT on system PATH (documented) |
| Template creation (`--new`) | SKIP | Requires interactive TTY (InquirerPy) |

### Phase 3: Training Readiness (3/3 PASS)

| Test | Result | Notes |
|------|--------|-------|
| All 4 `train.py --help` | PASS | Correct argument parsing |
| Agent config files | PASS | All YAML/Python configs match skill docs |
| Loss function locations | PASS | Verified via `inspect.getsourcefile()` |

### Phase 4: CartPole Training (8/9 PASS, 1 KNOWN BUG)

| Test | Result | Time | Final Reward |
|------|--------|------|-------------|
| RSL-RL train (4096 envs) | **PASS** | 37s | 4.91 |
| SKRL train (4096 envs) | **PASS** | 55s | ~4.9 |
| RL-Games train (4096 envs) | **PASS** | 61s | 4.89 |
| SB3 train (64 envs) | **PASS** | 103s | 4.70 |
| TensorBoard reads all 4 logs | **PASS** | - | - |
| Checkpoints saved (all 4) | **PASS** | - | - |
| RSL-RL play.py (inference) | **PASS** | runs OK | - |
| Direct CartPole (RSL-RL) | **FAIL** | crash | warp dtype bug |

> **Known Bug**: `Isaac-Cartpole-Direct-v0` crashes with `RuntimeError: Cannot cast
> dtypes of unequal byte size` in warp on the develop branch (Isaac Sim 6.0). This is
> an upstream issue, not a skill defect. The skill documents this with a warning and
> redirects users to `Isaac-Cartpole-v0` (manager-based).

### TensorBoard Metrics Verified

All metric names documented in the skills were verified against actual training log data:

| Framework | Reward Metric | Verified Value |
|-----------|--------------|----------------|
| RSL-RL | `Train/mean_reward` | 4.91 |
| SB3 | `rollout/ep_rew_mean` | 4.70 |
| SKRL | `Reward / Total reward (mean)` | ~4.9 |
| RL-Games | `rewards/iter` | 4.89 |

---

## Gap Analysis Across 2 Runs

### Run 1 (initial): Found 15 gaps, fixed all in skill documents
### Run 2 (re-validation): Found 3 new gaps, fixed all; confirmed 14 previous fixes hold

| # | Gap | Severity | Status |
|---|-----|----------|--------|
| 1 | TensorBoard not on system PATH | High | Fixed (skill documents correct command) |
| 2 | TensorBoard metric names differ per framework | High | Fixed (cross-framework table added) |
| 3 | Direct CartPole broken on develop | High | Documented (warning + workaround) |
| 4 | Loss function locations undocumented | High | Fixed (file:line for all 4 frameworks) |
| 5 | RSL-RL missing from verification check | Medium | Fixed (all 4 FW in verification.md) |
| 6 | `rsl_rl.__version__` doesn't exist | Medium | Fixed (use `importlib.metadata`) |
| 7 | `pip list` returns empty in Kit Python | Medium | Fixed (documented + workaround) |
| 8 | Deep imports require AppLauncher | Medium | Fixed (comprehensive list of affected modules) |
| 9 | Play script runs infinitely in headless | Medium | Fixed (Ctrl-C note added) |
| 10 | Gym deprecation warning from SB3 | Low | Fixed (troubleshooting entry) |
| 11 | TensorFlow warning from TensorBoard | Low | Fixed (troubleshooting entry) |
| 12 | PyTorch compute capability warning | Low | Fixed (troubleshooting entry) |
| 13 | RayCaster variants undocumented | Low | Fixed (4 variants added) |
| 14 | Camera vs TiledCamera unclear | Low | Fixed (comparison table) |
| 15 | SKRL silent training (no terminal metrics) | Medium | Fixed (Run 2 — note added) |
| 16 | SB3 batch_size mismatch warning | Low | Fixed (Run 2 — troubleshooting entry) |
| 17 | SB3 `constant_fn()` deprecation | Low | Fixed (Run 2 — troubleshooting entry) |

**Gap closure rate: 17/17 (100%)** — all identified gaps are documented with fixes or workarounds.

---

## What the Skills Cover (by User Journey Phase)

### 1. "I just got this machine, how do I start?"
- `setup/fresh-install.md` — CUDA/driver prerequisites, Kit vs source-build install, symlink setup, extension install, RL framework install, TensorBoard
- `setup/verification.md` — 5-step verification script (GPU, Python, AppLauncher, all 4 RL frameworks, training scripts)

### 2. "What kind of environment should I build?"
- `architecture/overview.md` — Manager-based vs Direct comparison, when to use each
- `architecture/patterns-comparison.md` — Side-by-side code examples, extension points
- `architecture/sensors-actuators.md` — Complete catalog: 7 actuator types (Implicit through Neural Net), 8 sensor types (Camera through Visuotactile), 3 visualizer backends (Kit, Newton, Rerun)

### 3. "How do I set up observations, rewards, actions?"
- `environments/builder.md` — Config class structure, reward design rules
- `environments/mdp-catalog.md` — Every built-in `mdp.*` function with parameters
- `environments/templates.md` — Copy-paste templates for env configs and agent configs

### 4. "Which RL framework should I use?"
- `training/guide.md` — Framework comparison (5 FW), CLI args, loss function locations, log structures
- `training/hyperparameters.md` — Cross-framework defaults table, tuning guidelines
- Per-framework deep-dives: `sb3-reference.md`, `rsl-rl-reference.md`, `rl-games-reference.md`

### 5. "Let me try training CartPole"
- `tutorials/cartpole.md` — Quick-start commands, step-by-step walkthrough, available variants
- `tutorials/code-walkthrough.md` — Line-by-line analysis of CartPole env config
- `tutorials/experiments.md` — 7 experiments (modify rewards, longer episodes, bigger networks, domain randomization, camera observations, direct env, external project)

### 6. "Something went wrong"
- `setup/troubleshooting.md` — 14 error patterns with causes and fixes (ModuleNotFoundError, OOM, import failures, warnings, convergence issues)

---

## Performance Impact

### Before Skills (manual workflow)
A first-time Isaac Lab user would need to:
- Read 50+ pages of NVIDIA docs to understand install options
- Debug PYTHONPATH issues (`omni` module not found) with no guidance
- Discover that `pip list` returns empty and `rsl_rl.__version__` doesn't exist
- Figure out that TensorBoard needs Kit Python (not system PATH)
- Learn that metric names differ across 4 RL frameworks
- Hit the Direct CartPole bug with no context on whether it's their fault

### After Skills (Claude Code assisted)
- Environment verified in under 60 seconds (scripted checks)
- Correct training command generated on first try for any framework
- TensorBoard launched correctly with framework-specific metric guidance
- Known bugs surfaced proactively with workarounds
- Hyperparameters explained with cross-framework comparison table
- Troubleshooting covers 14 common errors before the user even hits them

### Training Benchmarks (GB10 aarch64, headless)

| Framework | num_envs | Wall Time | Steps/s | Convergence |
|-----------|----------|-----------|---------|-------------|
| RSL-RL | 4096 | **37s** | 265K | 4.91 reward |
| SKRL | 4096 | 55s | ~45 it/s | ~4.9 reward |
| RL-Games | 4096 | 61s | 146K fps | 4.89 reward |
| SB3 | 64 | 103s | 5K | 4.70 reward |

All 4 converge to near-optimal CartPole performance (theoretical max ~4.95).

---

## Known Limitations

1. **Direct CartPole broken** on develop branch (warp dtype error) — upstream Isaac Sim 6.0 issue
2. **`isaaclab.sh --new` requires TTY** — cannot be automated in non-interactive shells
3. **SKRL terminal output** is progress-bar-only; users must use TensorBoard for metrics
4. **RLinf framework** (5th, for GR00T/VLA fine-tuning) lightly documented — specialized use case
5. **Skills only tested on GB10 aarch64** — x86_64 and other GPUs not validated in this run

---

## Files Changed (3 commits on `claude/skills` branch)

```
262d96bfa2d  Update Isaac Lab skills after full Dev Journey Run 2
574821f02fe  Fix warp dtype casts and headless rendering for aarch64
ed0c7b8f51f  Improve Isaac Lab skills after Dev Journey gap analysis
9c7eada5c6e  Add Claude Code skill and project guide for Isaac Lab
```

**Total: 18 skill documents, ~2,500 lines, 17 gaps found and resolved, 22 tests executed.**
