# Cosmos Transfer demo setup and training

Run the Shadow Hand camera task with Cosmos3-Nano-Sim-Transfer using the setup below.
The revisions below record the demo's tested configuration. The Framework checkout
is private and requires repository access; the checkpoint requires access to
[nvidia/Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer).

Use **uv 0.11.29** for the Framework environment. Keep the Cosmos Framework and
Isaac Lab environments separate. The demo requires **48 GB of VRAM for one
environment**, with approximately **8 GB more for each additional environment**.
Choose hardware with enough VRAM before increasing `--num_envs`.

For service options and camera configuration, see [Serving Cosmos](cosmos_service.md)
and [Cosmos camera integration](cosmos.md).

## 1. Set paths

Start in a parent directory that will contain `IsaacLab`, `cosmos-framework`, and
`model`. Set these variables before cloning:

```bash
export ISAAC_LAB="${PWD}/IsaacLab"
export COSMOS_FRAMEWORK="${PWD}/cosmos-framework"
export COSMOS_CHECKPOINT="${PWD}/model"
```

In each new terminal, set the same three variables to those absolute paths before
changing directories. Repeating the block from a different directory would select
different paths.

## 2. Clone the tested revisions

Clone the experimental Isaac Lab branch and select the demo revision:

```bash
git clone -b exp/cosmos https://github.com/isaac-sim/IsaacLab.git "$ISAAC_LAB"
git -C "$ISAAC_LAB" fetch origin
git -C "$ISAAC_LAB" checkout 4f8fb991f
```

Clone the private Cosmos Framework variant and select its matching revision:

```bash
git clone https://github.com/nvidia-cosmos/cosmos-framework-private.git "$COSMOS_FRAMEWORK"
git -C "$COSMOS_FRAMEWORK" fetch origin
git -C "$COSMOS_FRAMEWORK" checkout be540e475
```

## 3. Install dependencies and download the checkpoint

From the Framework checkout, create its environment and install Isaac Lab's Cosmos
worker into it:

```bash
cd "$COSMOS_FRAMEWORK"
uv sync --locked --all-extras --group=cu130-train
uv run --no-sync uv pip install -e \
  "$ISAAC_LAB/source/isaaclab_experimental/isaaclab_experimental/cosmos"
```

Download the complete checkpoint. This command uses the Framework environment's
Hugging Face CLI; if authentication is required, run `uv run --no-sync hf auth login`
with an account that has model access first.

```bash
uv run --no-sync hf download nvidia/Cosmos3-Nano-Sim-Transfer \
  --local-dir "$COSMOS_CHECKPOINT"
```

Keep `--no-sync` on subsequent Framework commands to preserve the separately
installed worker. If you run `uv sync` again, repeat the worker installation.

## 4. Start the Cosmos service in terminal 1

With the path variables set:

```bash
cd "$COSMOS_FRAMEWORK"
export LD_LIBRARY_PATH=
export PYTHONFAULTHANDLER=1
uv run --no-sync isaaclab-cosmos-server --checkpoint "$COSMOS_CHECKPOINT" 2>&1 | tee server.log
```

Wait for `Cosmos ready at ...` before starting training. Leave this terminal
running; its output is also saved in `server.log` in the Framework checkout.

## 5. Start Isaac Lab training in terminal 2

Set the path variables to the same absolute paths as in terminal 1, then run:

```bash
cd "$ISAAC_LAB"
uv run isaaclab cosmos status
```

Check for `"ready": true` and `"session_active": false`, then start the demo:

```bash
uv run isaaclab train \
  --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
  --rl_library rsl_rl \
  --max_iterations 20 \
  --num_envs 1 \
  presets=cosmos
```

The `cosmos` preset sends camera depth to the service and uses the generated RGB
images for training. This command runs 20 training iterations with one environment.

## 6. Optional: record a video

After the previous training process exits, run this command in the Isaac Lab
terminal to train again while recording 300 steps of the Newton GL streaming view:

```bash
cd "$ISAAC_LAB"
uv run --extra video isaaclab train \
  --task Isaac-Reorient-Cube-Shadow-Camera-Direct \
  --rl_library rsl_rl \
  --max_iterations 20 \
  --num_envs 1 \
  --viz newton_gl \
  --video viz:newton_gl:streaming_view \
  --video_length 300 \
  presets=cosmos
```

Without a screen, run `unset DISPLAY` before the recording command. The training
process prints the recording's output directory. Press Ctrl+C in terminal 1 when
finished to stop the Cosmos service and release the model.
