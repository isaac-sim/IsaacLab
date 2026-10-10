# GR00T N1.7 with native skrl PPO

This local runner fine-tunes the **complete original N1.7 action head** from the existing micro-SFT checkpoint. It uses skrl 2.1.0 PyTorch `PPO`, `RandomMemory` and `SequentialTrainer` for rollout collection, GAE, losses, backward passes, Adam updates, TensorBoard and checkpoint serialization. The frozen vision/language backbone is held separately from the policy, optimizer and checkpoint.

The simulator and model run in separate uv environments and processes. The simulator uses Isaac Sim 6.1 / PhysX, one Franka and the task's table/wrist RGB cameras. This is an engineering smoke test with a local approach reward; it does not establish stacking ability or PPO convergence. Shared task definitions and Isaac-GR00T source are unchanged.

## Debugging and learning

Open the Isaac Lab root in VS Code with the Python and Python Debugger extensions installed on
the remote host. Select **GR00T PPO: 两步训练（自动调试模型与仿真）** in
[launch.json](../../../.vscode/launch.json) and press F5. The configuration automatically attaches
both Python children, runs one two-step rollout and extends RPC deadlines to two hours for
breakpoints. Resume and inference configurations prompt for a native checkpoint's absolute path.
Each launch creates a fresh output directory.

The debug-only `--debug_subprocesses` option invokes each project's existing `.venv/bin/python`
directly, allowing VS Code's `subProcess` injection to reach both environments without following
uv's Rust process. It does not install dependencies or change the training loop. Normal launches
still use `uv run --no-sync`. Child output remains in `train.log` and `simulation.log`.

See the Chinese [RL pipeline learning guide](../../../docs/RL_PIPELINE_GUIDE.md) for the theory,
code map, breakpoint walkthrough, tensor shapes and checkpoint workflow.

## Installation

Run from the Isaac Lab checkout. The existing environments are `.venv` and `../Isaac-GR00T/.venv`. Preserve the existing model dependencies when adding skrl:

```bash
uv --no-config pip freeze --python ../Isaac-GR00T/.venv/bin/python \
  --exclude-editable > /tmp/gr00t-skrl-existing-constraints.txt
uv --no-config pip install --python ../Isaac-GR00T/.venv/bin/python \
  --constraint /tmp/gr00t-skrl-existing-constraints.txt \
  -r scripts/reinforcement_learning/gr00t_skrl/requirements-model.txt
```

`--no-config` is necessary here: Isaac Lab's uv configuration overrides NumPy to version 2+, whereas the isolated GR00T environment requires NumPy 1.26.4. Runtime commands use `--no-sync` so that the two environments keep their separate dependency versions.

Validated model versions: Torch 2.9.0+cu128, NumPy 1.26.4, Transformers 4.57.3, skrl 2.1.0 and TensorBoard 2.20.0. Simulator versions are recorded separately with each run. The default inputs are:

- `../embodied-template/models/stack-cube-n1.7-sft/checkpoint`
- `../embodied-template/models/nvidia/Cosmos-Reason2-2B`

Keep `nvidia/Cosmos-Reason2` in the backbone path, including its symlink name; the upstream model selects its backbone class from this string. Model assets must already be local. The launcher sets `HF_HUB_OFFLINE=1` and the previously accepted `OMNI_KIT_ACCEPT_EULA=YES`.

## One-command cloud deployment

The deployment script targets **Ubuntu 22.04/24.04 x86_64** with a rendering-capable NVIDIA GPU
(24 GB VRAM or more), at least 32 GB host RAM and driver **580.65.06 or newer** for the simulator's
CUDA 13.0 Torch build. Driver 580.95.05+ is recommended by the Isaac Lab installation guide.
CUDA availability alone does not establish Isaac Sim rendering support; use a GPU supported by
[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
Allow at least **100 GiB free** for environments, download caches and smoke-test outputs in addition
to the transferred models; each PPO checkpoint adds about 9.1 GiB. The script checks the checkout,
GR00T and uv cache volumes. It uses the provider's installed driver; CUDA runtime libraries come
with the Python wheels, so a separate CUDA toolkit is unnecessary for this installation.

Transfer this Isaac Lab branch and both model directories to the server first. The micro-SFT
checkpoint is a local training artifact; the script does not substitute a public base model.
Do **not** copy `.venv` directories between servers. When transferring the backbone, dereference
its symlink (for example, use `rsync -aL`) or also transfer its target. Keep the destination name
`nvidia/Cosmos-Reason2-2B`. With the default sibling directory layout:

```text
workspace/
├── IsaacLab/                 # this branch, including uv.lock and deploy.sh
├── Isaac-GR00T/              # script creates this if absent
└── embodied-template/models/
    ├── stack-cube-n1.7-sft/checkpoint/
    └── nvidia/Cosmos-Reason2-2B/
```

Run from the transferred Isaac Lab checkout:

```bash
bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh
```

It installs OS packages with root/passwordless sudo, bootstraps uv if absent, obtains Python 3.12,
fetches the validated GR00T commit `d2b7e75b937e3ec9aa5dbc798f08b89692c49734`, and installs the
two `.venv` environments from their frozen lockfiles. GR00T sync runs from its own directory;
the additional PPO dependencies use `uv --no-config pip` with a snapshot constraint file.
Existing extra packages are retained by `--inexact`. A different or modified existing GR00T
checkout is rejected; pass a new `--model_project` to keep it separate. The deployment is
noninteractive and uses the previously accepted `OMNI_KIT_ACCEPT_EULA=YES`.

Default verification checks dependency versions, both CUDA environments, local weight shards,
offline processor loading and a falling procedural sphere in headless Kit/PhysX using the task's
simulation config. It prints the launch command with your paths and GPU. The processor uses
`--backbone_path` even when its saved config contains a previous server's absolute path; saved
checkpoint files and processor fingerprints remain unchanged.

For custom locations, a preconfigured image, and full two-step PPO verification:

```bash
bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh \
  --model_project /data/Isaac-GR00T \
  --model_path /data/models/stack-cube-n1.7-sft/checkpoint \
  --backbone_path /data/models/nvidia/Cosmos-Reason2-2B \
  --gpu 0 --skip_system_deps --smoke_test
```

`--skip_system_deps` requires the OS libraries listed in the script to be installed already.
`--smoke_test` exercises the real Franka scene, both RGB cameras, two rollout steps, native PPO
updates and checkpoint writing. It requires network access to the task's USD assets or an existing
asset cache. Installation also needs access to Ubuntu package repositories, GitHub, PyPI, NVIDIA
and PyTorch wheel indexes. Set your proxy and `UV_CACHE_DIR` before running when needed; frozen
lockfiles retain their recorded download URLs.

Other useful invocations:

```bash
# Inspect the deployment plan without installing or launching anything.
bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh --dry_run
# Recheck existing environments without reinstalling.
bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh --verify_only
```

Pass the same custom path/GPU arguments when verifying. Logs and model dependency snapshots are
written under `logs/gr00t_skrl/deploy/<timestamp>/`; optional smoke outputs live in its `smoke/`
subdirectory. Re-running resumes package installation without rewriting lockfiles or overwriting
prior run directories. A failed check exits nonzero and prevents reporting deployment success.

## Policy and physical action contract

For the model-configured horizon `H`, padded action dimension `D`, generation steps `K` and fixed `sigma=0.05`:

```text
x_0 ~ Normal(0, I)
x_(k+1) ~ Normal(x_k + velocity_theta(x_k, observation, k) / K, sigma)
log_prob = sum over k, H and D of log Normal(x_(k+1); conditional_mean_k, sigma)
```

The parameter-independent density of `x_0` is omitted; it cancels in the PPO ratio. This is the joint transition density of the complete generation chain. It includes padded coordinates and unexecuted horizon positions. It is not the marginal density of the robot command.

Native memory stores flattened `x_0 ... x_K`. Teacher-forced likelihood uses these exact saved actions. The current model has `H=40`, `D=132`; `K=2` therefore stores 15,840 FP32 scalars per action. Dimensions are read from the model config. Dropout is disabled in both sampling and learning, with autograd retained. Backbone features, masks and normalized processor state are padded into a fixed observation (`--max_tokens`, default 1024); overflow raises an error. Trainable state encoders, VL normalization/self-attention and projectors are recomputed during learning.

Each environment step predicts a new chain, decodes `x_K` using the original processor and executes only its first seven-dimensional command. Arm values are clipped to `[-0.1, 0.1]`, then passed to the unchanged task action with `scale=0.5`; the runner does not multiply them again. The gripper command is binary. The processor's legacy `roll/pitch/yaw` state names contain a **principal rotation vector [rad]**, alongside XYZ [m] and two finger positions [m]. The task's signed second-finger coordinate is preserved, matching the micro-SFT processor statistics.

The local native reset event restores initial scene states and joint targets, disabling randomization only in this runner. The local reward is `step_dt * (1 - tanh(distance_to_red_cube / 0.1))`, using actual scene geometry; native reward-manager integration supplies `step_dt`. Task success, failure and timeout terms are inherited.

The state critic is `[8, 64, 1]`. With `compute_final_obs=True`, a timeout passes the reset-before-terminal critic state to native PPO bootstrapping. Simultaneous true termination and timeout does not bootstrap. The next interaction uses normal reset-after-terminal observations. The wrapper consumes the native trainer's single-environment reset request without resetting the already-reset simulation a second time.

## Training, checkpoint and inference

The real two-step smoke command succeeded on the local RTX 4090:

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
  --run_dir logs/gr00t_skrl/smoke_v3 --episode_steps 2
```

It produced `logs/gr00t_skrl/smoke_v3/native/checkpoints/agent_2.pt`, completed four native Adam steps, changed the original head, preserved the frozen backbone and measured end-effector displacements of approximately 9.9 mm and 14.7 mm. Likelihood recomputation error was zero. Total GPU memory sampled once per second peaked at 20,796 MiB. Initial sampling / native update / post-update backbone restore peaked at 5.97 / 12.88 / 15.00 GiB of model-process PyTorch allocations.

The smoke used learning rate `1e-6`: post-update joint-chain ratios ranged from 0.051 to 8.15 (approximate KL 3.54, clip fraction 1.0). Losses and gradients were finite. These numbers demonstrate an update, **not a tuned or converged policy**; sustained training needs a smaller learning-rate experiment and a task reward appropriate for stacking.


The following fresh-process resume also succeeded:

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
  --run_dir logs/gr00t_skrl/resume --episode_steps 2 \
  --resume logs/gr00t_skrl/smoke_v3/native/checkpoints/agent_2.pt
```

It restored head, critic and all native optimizer state tensors exactly, then advanced environment steps `2 -> 4`, updates `1 -> 2` and optimizer steps `4 -> 8`. It wrote `logs/gr00t_skrl/resume/native/checkpoints/agent_4.pt`. Model-process loading peaked at 9.08 GiB of PyTorch allocations; total sampled GPU memory peaked at 20,894 MiB. The second update retained finite losses/gradients (ratios 0.492–2.24, approximate KL 0.317). Both children exited with code zero and their sockets were removed.

Fresh-process inference succeeded without additional optimizer steps:

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
  --run_dir logs/gr00t_skrl/inference --mode inference --episode_steps 2 \
  --resume logs/gr00t_skrl/resume/native/checkpoints/agent_4.pt
```

It kept optimizer step 8, validated both RGB cameras and measured approximately 11 mm of motion on each step. CPU loading peaked at 16.43 GiB RSS, model-process loading at 9.08 GiB of PyTorch GPU allocations, and the complete inference run at 17,848 MiB of sampled total GPU memory. Outputs and checkpoints are ignored artifacts, not source files. Choose a fresh `--run_dir` to reproduce these commands, or omit that option for an automatically named directory.

The launcher defaults to two environment steps, two learning epochs and two minibatches of size one (four native optimizer steps). `--episode_steps 2` explicitly exercises timeout and automatic reset. Run directories must be new; existing output is never overwritten.

A boundary checkpoint contains the original head, critic, native Adam state and a `RunState` module with cumulative counters, RNG, dependency versions, processor fingerprints and the policy contract. Automatic best-checkpoint copying is disabled. The lifecycle hook uses native `save` at full rollout boundaries and native `load` after CPU compatibility validation. CPU loading avoids a duplicate GPU copy of all checkpoint tensors. Policy, critic and optimizer tensors are independently fingerprinted before/after loading. RNG is restored after initialization and diagnostic inference; resumed physics starts from a new episode.

The head uses BF16, with FP32 chain dynamics, densities, ratios and losses. Local per-instance forward wrappers checkpoint the upstream DiT and VL transformer blocks without changing parameters or state-dict names. The backbone moves to CPU during native learning and checkpoint loading, then returns to the GPU. Native Adam uses `foreach=False`. Entropy bonus, adaptive learning rate and extra observation/value normalization are disabled. Native "Policy / Standard deviation" describes the fixed conditional transition noise.

Each output directory contains `config.json`, separate simulator/model logs, `native/` TensorBoard files and checkpoints, `metrics.json` and `processes.json`. Diagnostics include same-chain likelihood error, native losses, post-update ratios/KL/clip fraction, finite gradients, changed-head and frozen-backbone fingerprints, optimizer counters, executed controls, measured motion, RGB variation and memory peaks. PyTorch peaks describe the model process; `processes.json` samples total GPU memory once per second, including simulation. RPC has bounded messages and deadlines; child failure or interruption triggers process-group cleanup and private socket removal.

## Contract tests

These tests run entirely in the model environment; the parent test process never starts Kit:

```bash
NO_ALBUMENTATIONS_UPDATE=1 uv run --project ../Isaac-GR00T --no-sync python -m pytest \
  scripts/reinforcement_learning/gr00t_skrl/tests/test_contracts.py -q \
  --confcutdir=scripts/reinforcement_learning/gr00t_skrl/tests
```

Coverage also checks real NumPy 2.x ↔ 1.26 RPC compatibility, EOF and read deadlines. Coverage protects full-chain probability/memory, original-head gradient and dropout behavior, independently calculated timeout returns, and new-process restoration of model/optimizer/RNG/counters. Changing probability `sum` to `mean` or using reset state for timeout bootstrap makes the corresponding test fail. No native PPO/GAE implementation is copied.
