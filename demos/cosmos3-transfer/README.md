# Cosmos 3 Transfer on the Shadow Hand camera task

Isaac Lab renders the Shadow Hand camera's depth, Cosmos turns it into a realistic image, and the policy camera sees
that image. Two terminals: one runs the Cosmos service, the other runs Isaac Lab. Everything below uses the built-in
commands from [Serving Cosmos](../../source/isaaclab_experimental/docs/cosmos_service.md) and
[Cosmos camera integration](../../source/isaaclab_experimental/docs/cosmos.md); read those for options.

Needs Linux, two GPUs (or MIG slices) of about 48 GB, and access to the Cosmos3-Nano-Sim-Transfer checkpoint.

## 1. Cosmos service (terminal 1)

```bash
export ISAAC_LAB_ROOT=/absolute/path/to/IsaacLab
export COSMOS_FRAMEWORK_ROOT=/absolute/path/to/cosmos-framework
export COSMOS_CHECKPOINT=/absolute/path/to/Cosmos3-Nano-Sim-Transfer

cd "$COSMOS_FRAMEWORK_ROOT"
uv sync --locked --all-extras --group=cu130-train
uv run --no-sync uv pip install --editable "$ISAAC_LAB_ROOT/source/isaaclab_experimental/isaaclab_experimental/cosmos"
uv run --no-sync hf download nvidia/Cosmos3-Nano-Sim-Transfer --local-dir "$COSMOS_CHECKPOINT"

export LD_LIBRARY_PATH=
uv run --no-sync isaaclab-cosmos-server --checkpoint "$COSMOS_CHECKPOINT" --kv-window 4
```

Wait for `Cosmos ready at ...`. Set up and download once; later runs only need the last two commands.

## 2. Isaac Lab (terminal 2)

```bash
cd "$ISAAC_LAB_ROOT"
uv run isaaclab cosmos status        # "ready": true, "session_active": false

uv run --extra video isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Direct --rl_library rsl_rl --num_envs 2 \
  --max_iterations 20 \
  --cosmos_prompt "A close-up overhead view of a robotic Shadow Hand turning a red cube on a wooden workbench." \
  --cosmos_prompt "A close-up overhead view of a robotic Shadow Hand turning a blue cube in a bright white lab." \
  --viz newton_gl --video viz:newton_gl:streaming_view --video_length 600 \
  presets=cosmos
```

The run prints the folder with the recorded video: the generated images of both environments side by side. Stop the
service with Ctrl+C in terminal 1.
