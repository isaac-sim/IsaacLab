# Cosmos runtime visual domain randomization

Runtime visual DR restyles the camera images consumed by a policy. The task supplies
RGB, depth, semantic segmentation, foreground classes, and prompts. The runtime
handles observation caching, episode styles, partial resets, action-chunk gating,
and model residency. This path generates independent images; it does not implement
the reference-frame video chains used in the offline experiments.

## Framework and checkpoint choices

Install either the public NVIDIA `cosmos-framework` or the GitLab
`mingxinz/guided_generation_distilled` branch. Framework and checkpoint are separate
choices:

| Framework | Public `nvidia/Cosmos3-Nano` | Custom four-step export | Mask guidance |
|---|---|---|---|
| Public NVIDIA | Supported | Supported when exported with its fixed-step config | Isaac Lab sampler adapter |
| GitLab guided-generation branch | Supported | Supported | Native guided-generation API |

Each combination also supports `mask_guidance: false`. Public transfer processes
samples sequentially, even when `max_batch > 1`; the GitLab implementation can pack
compatible samples. Remote workers provide parallelism across GPUs for either.

Install Isaac Lab and the `isaacsim` and `teleop` extras first. From the Isaac Lab
repository root, use one of:

```bash
# Public framework (the installer default):
bash scripts/visual_dr/install_cosmos.sh

# Existing git checkout
bash scripts/visual_dr/install_cosmos.sh /path/to/cosmos-framework

# Or a Git source pinned to the branch (substitute your GitLab URL):
bash scripts/visual_dr/install_cosmos.sh \
  'ssh://<path>/cosmos-framework.git@<branch name>'
```

The installer adds the `isaaclab_contrib[cosmos-runtime]` dependencies and installs
Cosmos with `--no-deps` to retain Isaac Lab's torch version. Cosmos is an external
overlay because its dependency constraints conflict with the workspace's newer
transformers version. Use `uv sync --inexact` to retain the overlay, or reinstall it
after an exact sync. Attention extensions must match the environment's Python,
PyTorch, and CUDA versions; do not install an unrelated FlashAttention wheel.

## Public Nano recipe

Start with `scripts/visual_dr/recipes/cosmos_nano.yaml`. It selects the public
checkpoint, 50 UniPC steps, text guidance 3, segmentation control guidance 2, and
the standard 720 / 4:3 bucket (1104x832). Camera capture can remain 640x480; the
backend resizes controls to the generation bucket and resizes the result back.
It does not override the public bucket with 640x480.

```bash
uv run --inexact python scripts/visual_dr/run_rollout.py \
  --backend cosmos --num_envs 1 --steps 8 \
  --recipe scripts/visual_dr/recipes/cosmos_nano.yaml

# Same checkpoint and controls, with mask guidance disabled:
uv run --inexact python scripts/visual_dr/run_rollout.py \
  --backend cosmos --num_envs 1 --steps 8 \
  --recipe scripts/visual_dr/recipes/cosmos_nano.yaml --no-mask_guidance
```

`run_rollout.py` accepts `--recipe`; explicit CLI settings apply after the recipe.
`apply_visual_dr_recipe()` can also apply the YAML to a task's `VisualDRCfg` in
Python. It retains task prompts, semantic classes, and
camera names. Unknown fields are rejected.

`checkpoint` accepts `nvidia/Cosmos3-Nano`, registered Cosmos names, local export
directories, and S3 URIs. The Hub ID resolves to the framework's public Nano
registration. A local directory must contain the checkpoint configuration and
weights in Cosmos export format. Older local exports missing the four modality
embedding flags are loaded through a temporary config view with legacy defaults;
the original config and weight files are unchanged.

## Custom checkpoints

Use `--checkpoint` to select a custom or distilled Cosmos export, with sampling
and resolution settings appropriate for that checkpoint. For example, run a
four-step transfer export at the camera resolution:

```bash
uv run --inexact python scripts/visual_dr/run_rollout.py \
  --num_envs 1 --steps 100 --episode_length_s 30 --policy scripted \
  --backend cosmos --checkpoint /path/to/four-step-transfer-checkpoint \
  --num_steps 4 --guidance 1.0 --control_guidance 1.0 --resolution 480 \
  --native_resolution --mask_guidance --mask_strength 1.0 \
  --video_dir /tmp/cosmos_rollout
```

You can also use `scripts/visual_dr/recipes/cosmos_distilled.yaml` as a starting
recipe and override its checkpoint or settings on the command line.

Boundary dilation defaults to zero and foreground compositing defaults to off.
With mask guidance enabled, `mask_step_threshold: null` (the default) projects the
preserved source through the final denoising step. No extra flags are needed for
these defaults.

## Guidance, boundaries, and compositing

The recipes enable mask guidance and disable foreground compositing. At guided
updates, the source RGB is encoded and its noisy latent is blended into the
preserve region. The model still decodes the foreground. On public Cosmos, an
adapter wraps the selected sampler for one request and restores it afterward;
it does not copy a sampler implementation or paste source pixels into the output.

- `mask_step_threshold: null` applies guidance through every update: index 49 for
  the public 50-step recipe and index 3 for the four-step recipe. An explicit
  smaller index releases the mask for later updates.
- `mask_strength` is a blend applied each guided update, not an overall percentage
  of final preservation. Repeated blending can make nearby strengths look similar.
  Strong guidance retains source lighting as well as geometry; softer guidance
  allows relighting but also geometry and color changes.
- `mask_downsample_mode: area` retains fractional boundary cells. Max pooling can
  retain source background around an object's silhouette and cause a bright fringe.
- `boundary_erosion_px: 8` shrinks the **union** of preserved masks in camera pixels.
  Shared edges stay filled. Adjust the radius for the camera resolution and thin
  objects. A mapping such as `{table: 8}` instead shrinks only exposed table edges.
  Positive erosion cannot be combined with positive `boundary_px` dilation.
- `composite_foreground` defaults to false. Setting it to true explicitly pastes
  source pixels back after generation.
  It preserves exact pixels and their original lighting. It is independent of
  `mask_guidance`; disabling guidance does not itself disable compositing.

Keep `on_error: raise` while validating. Passthrough on failure is opt-in and records
the error; it should not be mistaken for a successful guided result.

## Runtime integration and checks

`FrankaStackRuntimeDRCfg` in `isaaclab_tasks` connects rendered signals to the visual
DR observation term. The runtime is attached and activated after environment
construction, avoiding model loads during observation-shape probes. Call `offload()`
before a memory-heavy learner update and `activate()` before collection resumes.
Training integrations must call these lifecycle methods explicitly. This branch
provides the runtime and demo wiring; it does not add an RLinf training adapter.

`--workers 1,2` moves generation into separate same-node processes. The simulator
and worker GPUs must all be visible; inputs use CUDA IPC and peer copies. Worker
configuration retains the checkpoint, mask, resolution, and sampler settings.

Run the CPU contract tests with the Cosmos overlay installed:

```bash
uv run --inexact python -m pytest source/isaaclab_contrib/test/visual_dr -q
```

The tests cover scheduling, mask construction, compositing, recipes, and sampler
projection. GPU inference and simulator rollouts remain separate integration checks.
