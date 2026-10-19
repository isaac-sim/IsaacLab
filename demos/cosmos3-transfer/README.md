# Cosmos 3 Transfer on the Shadow Hand camera task

Isaac Lab renders the Shadow Hand camera's depth, Cosmos turns it into a realistic image, and the policy trains on that
image. Run the Cosmos service in one terminal and `isaaclab train` in another. These are the commands from
[Serving Cosmos](../../source/isaaclab_experimental/docs/cosmos_service.md) and
[Cosmos camera integration](../../source/isaaclab_experimental/docs/cosmos.md); see those pages for every option.

## 1. Start the Cosmos service

In its own terminal, with the paths of your Isaac Lab checkout, the
[Cosmos Framework](https://github.com/NVIDIA/cosmos-framework) checkout and the checkpoint folder:

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

The first three commands are one-time setup. Wait for `Cosmos ready at ...`.

## 2. Train with Cosmos on the camera

In a second terminal, train with the Cosmos preset and one prompt per environment:

```bash
cd "$ISAAC_LAB_ROOT"
uv run isaaclab cosmos status      # expect "ready": true and "session_active": false

uv run --extra video isaaclab train --task Isaac-Reorient-Cube-Shadow-Camera-Direct --rl_library rsl_rl --num_envs 2 \
  --max_iterations 20 \
  --viz newton_gl --video viz:newton_gl:streaming_view --video_length 600 \
  presets=cosmos \
  env.scene.tiled_camera.modifiers.distance_to_image_plane.1.backend.prompt="['A close-up overhead view of a robotic Shadow Hand turning a red cube on a wooden workbench.', 'A close-up overhead view of a robotic Shadow Hand turning a blue cube in a bright white lab.']"
```

- `presets=cosmos` selects the task's Cosmos camera: depth goes to the service and the generated image replaces the
  camera's `rgb`.
- The last argument is an ordinary Isaac Lab override of the Cosmos step in the camera's modifier chain. Its `prompt`
  describes the scene of each environment: with `--num_envs 2` the first prompt is environment 0 and the second is
  environment 1. The same path sets `endpoint` and `transport`, and `...distance_to_image_plane.0.params.near` and
  `far` set the depth range; see the [camera integration guide](../../source/isaaclab_experimental/docs/cosmos.md).
- `--viz newton_gl --video viz:newton_gl:streaming_view --video_length 600` records 600 steps (one episode) of the
  generated images of both environments side by side. The run prints the folder with the clip.
- `--kv-window 4` on the service lets two environments fit on a 48 GB GPU that Isaac Sim shares; omit it on a larger GPU.

Press Ctrl+C in the first terminal to stop the service.
