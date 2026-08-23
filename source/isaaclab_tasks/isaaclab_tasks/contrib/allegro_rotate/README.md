# Allegro Rotate IsaacLab Task

This package contains the Allegro in-hand cylinder-rolling task and its grasp-cache generator.

## Demo

![Allegro rotate demo](docs/allegro_rotate_demo.gif)

## 1. Setup

This task is self-contained in an upstream Isaac Lab checkout under:

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run all commands from the Isaac Lab root:

```bash
cd <ISAACLAB_ROOT>
```

The commands below use `uv run --extra isaacsim`, which creates and manages the
Python environment automatically. The first invocation downloads Isaac Sim and
its dependencies; no manual environment activation is required.

## 2. Grasp Cache

A pre-generated cache is available at
[`cache/allegro_grasp_linspace_0.8-0.8-1.npy`](cache/allegro_grasp_linspace_0.8-0.8-1.npy).
It is stored at
`source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate/cache/allegro_grasp_linspace_0.8-0.8-1.npy`
relative to the Isaac Lab root. Use that task-local path to skip grasp-cache generation.

Generate a new cache only when changing the cylinder scale range, hand initialization, or grasp criteria:

```bash
uv run --extra isaacsim python \
  source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate/tools/allegro_gen_grasp.py \
  --task IsaacContrib-Inhand-Rotate-Grasp-Allegro-v0 \
  --num_envs 4096 \
  --max_cache_rows 50000 \
  --output source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate/cache/allegro_grasp_linspace \
  --viz none
```

Expected cache:

```text
source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate/cache/allegro_grasp_linspace_0.8-0.8-1.npy
```

## 3. Visualize Cache

```bash
uv run --extra isaacsim python \
  source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate/tools/allegro_viz_cache.py \
  --cache source/isaaclab_tasks/isaaclab_tasks/contrib/allegro_rotate/cache/allegro_grasp_linspace_0.8-0.8-1.npy \
  --num_envs 1 \
  --viz kit \
  --real-time
```

## 4. Train Policy

Use the generated grasp cache to train the rolling policy with IsaacLab's RSL-RL training script:

```bash
uv run --extra isaacsim isaaclab train \
  --rl_library rsl_rl \
  --task IsaacContrib-Inhand-Roll-Allegro-v0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --viz none
```

## 5. Visualize Trained Policy

Find recent checkpoints:

```bash
find logs/rsl_rl/allegro_inhand_roll -name "model_*.pt" -printf "%T@ %p\n" \
  | sort -n \
  | tail -5
```

Validate every candidate checkpoint at the final training gravity. This disables the
gravity curriculum and starts the environment at `-10 m/s²`:

```bash
uv run --extra isaacsim python scripts/reinforcement_learning/play.py \
  --rl_library rsl_rl \
  --task IsaacContrib-Inhand-Roll-Allegro-v0 \
  --num_envs 16 \
  --viz kit \
  --checkpoint <CHECKPOINT_PATH> \
  env.gravity_curriculum=false \
  env.sim.gravity="[0.0, 0.0, -10.0]"
```

## 6. Reference

- [Sharpa RL Lab](https://github.com/sharpa-robotics/sharpa-rl-lab)
