# Local GR00T N1.7 PPO pipeline

This small runner connects one Isaac Sim 6.1 / PhysX Franka three-cube stacking
environment to the official N1.7 model in a separate uv environment. It uses
PyTorch directly and does not import or execute RLinf. Its purpose is to verify
sampling, action-head updates, checkpoint saving, and fresh-process resume on
one RTX 4090. It does not establish stacking ability or training convergence.

## Environments and input checkpoint

Run the launcher from the Isaac Lab checkout. The Isaac Lab environment needs
the `isaacsim` and `teleop` extras: the existing stacking configuration imports
the teleoperation configuration package even when no teleoperation is used.
The model process uses the `.venv` in `--gr00t_project`. Both processes are
launched with `uv run --no-sync`, preserving their separate dependency versions.
Follow each repository's installation instructions before using this runner.

Provide an N1.7 checkpoint and processor configured for the stack-cube
observation/action contract. The locally available micro-SFT checkpoint is a
usable engineering input; no SFT quality is assumed. The bare pretrained model
does not contain this task's processor configuration. Official demonstration
replay and Mimic expansion are separate data-preparation work, not prerequisites
for this smoke run when a compatible checkpoint already exists.

The current processor contract uses `libero_sim`: two RGB views (`image` and
`wrist_image`), XYZ and principal rotation vectors, and two gripper joint positions
(the second position is negated by the existing environment observation). Actions
contain six relative IK commands and one binary gripper command. The processor
normalizes these commands as seven scalar `NON_EEF` channels; they are decoded
into the existing environment's action coordinates, not absolute poses. The
processor's legacy `roll`, `pitch`, and `yaw` names refer to rotation-vector
components; they do not specify Euler angles. Isaac Lab applies its existing
arm-action scale of 0.5. This is a task adapter, not a
claim that LIBERO and Isaac Lab controllers are interchangeable.

`--backbone_path` must be a local Cosmos checkpoint whose path contains
`nvidia/Cosmos-Reason2`, as required by the official N1.7 backbone selector.
For example, a `nvidia/Cosmos-Reason2-2B` symlink may point to the downloaded
backbone directory. The runner preserves that path rather than resolving the
symlink. It preserves the action head's checkpoint architecture, padded action
dimensions, and prediction horizon.

## Run and resume

After accepting the NVIDIA Omniverse EULA, set `OMNI_KIT_ACCEPT_EULA=YES`.
For the existing local assets, run from the Isaac Lab root:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
uv run --no-sync python scripts/reinforcement_learning/gr00t/local_pipeline.py \
  --gr00t_project ../Isaac-GR00T \
  --model_path ../embodied-template/models/stack-cube-n1.7-sft/checkpoint \
  --backbone_path ../embodied-template/models/nvidia/Cosmos-Reason2-2B \
  --output_dir logs/gr00t/train \
  --rollout_steps 2 --denoising_steps 2 --updates 1
```

Use `--updates 0` for a closed-loop inference run. To resume in new processes,
repeat the command with `--checkpoint logs/gr00t/train/checkpoint.pt` and a new
`--output_dir logs/gr00t/resume`. The optimizer step must advance from 1 to 2.
The default rollout length is eight environment steps; the two-step command
above is the smallest update check. One predicted action is executed per call.

Outputs include `sim.log`, initial camera PNGs, `simulation.json`, `metrics.json`,
and `checkpoint.pt`. The checkpoint stores action-head and value-network weights,
AdamW state, and RNG states; it still depends on the original model and processor.
Each full-head checkpoint is several GB. Resume starts a new simulator episode,
not the previous episode's exact physical state.

## PPO probability contract

The actor is a **stochastic denoising chain**, extending the original action head
without replacing it with a Gaussian MLP or residual policy. For every denoising
step it samples `x_next ~ Normal(x + velocity_theta(x) / K, 0.05)`. The initial
standard-normal latent is independent of model parameters and cancels from the
PPO ratio. This fixed-noise transition is an explicit smoke-training policy,
not an implementation of a particular published flow-SDE algorithm.

The rollout stores the complete latent chain and frozen backbone features. PPO
recomputes the joint transition log probability on the same chain and clips the
joint probability ratio to `[0.8, 1.2]`. It sums over all predicted dimensions
and timesteps, including unexecuted latents; those latents remain part of the
extended policy. This is not the marginal probability of the final physical
action. Decoding, clipping the six arm commands to `[-0.1, 0.1]`, and binary
gripper control form a deterministic mapping from that extended action to the
environment command. Do not mask dimensions or average log probabilities while
claiming to preserve this joint probability definition.

The visual/language backbone is frozen. The complete original action head and
a small state-based value network are trained. Dropout remains disabled during
both sampling and likelihood recomputation. The frozen backbone moves to CPU
during updates, and DiT evaluation is checkpointed to reduce activation memory.
The actor uses BF16 parameters/AdamW moments; probabilities and losses use FP32.
This memory-oriented smoke configuration is not a tuned numerical recipe.

The reward is the existing environment timestep multiplied by
`1 - tanh(distance_to_red_cube / 0.1)`, using actual simulation geometry. It is
only an approach reward, not a complete three-cube stacking reward. Termination
continues to use the existing three-cube success/failure rules. GAE bootstraps
timeouts from terminal observations and stops at both timeout and failure resets.

An update is accepted only when old log probabilities reproduce within 0.01,
loss and gradient norm are finite, decoder weights change, and the frozen
backbone receives no gradients and retains its weight SHA256. These checks
establish engineering behavior, not policy quality.

## Focused validation

```bash
NO_ALBUMENTATIONS_UPDATE=1 uv run --project ../Isaac-GR00T --no-sync python -m pytest \
  scripts/reinforcement_learning/gr00t/test_ppo_math.py -q \
  --confcutdir=scripts/reinforcement_learning/gr00t
```

The numerical test independently checks timeout bootstrapping and GAE isolation
across automatic resets. The real two-process training and resume runs validate
model gradients, simulation observations, and saved optimizer state.
