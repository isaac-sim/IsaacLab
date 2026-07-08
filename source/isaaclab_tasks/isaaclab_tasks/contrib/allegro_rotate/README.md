# Allegro Rotate IsaacLab Task

This package contains the Allegro in-hand cylinder-rotation task files copied in IsaacLab-relative structure.

## Demo

![Allegro rotate demo](docs/allegro_rotate_demo.gif)


## 1. Setup

Install IsaacLab first. For an upstream IsaacLab checkout, this task is self-contained under:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/allegro_rotate
```

For a standalone overlay copy, copy only the `source` tree into an IsaacLab repo:

```bash
cd <THIS_PACKAGE_ROOT>
rsync -a source <ISAACLAB_ROOT>/
```

The overlay does not replace IsaacLab's shared RSL-RL `train.py` or `play.py`.

Use the IsaacLab conda environment, then run all commands from:

```bash
cd <ISAACLAB_ROOT>
```

## 2. Generate Grasp Cache

```bash
./isaaclab.sh \
  -p source/isaaclab_tasks/isaaclab_tasks/direct/allegro_rotate/tools/allegro_gen_grasp.py \
  --task Isaac-Inhand-Rotate-Grasp-Allegro-v0 \
  --num_envs 4096 \
  --max_cache_rows 50000 \
  --output cache/allegro_grasp_linspace \
  --headless
```

Expected cache:

```text
cache/allegro_grasp_linspace_0.8-0.8-1.npy
```

## 3. Visualize Cache

```bash
./isaaclab.sh \
  -p source/isaaclab_tasks/isaaclab_tasks/direct/allegro_rotate/tools/allegro_viz_cache.py \
  --cache cache/allegro_grasp_linspace_0.8-0.8-1.npy \
  --num_envs 1 \
  --viz kit \
  --real-time
```

## 4. Train Policy

Use IsaacLab's existing RSL-RL training script:

```bash
./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Inhand-Rotate-Allegro-v0 \
  --num_envs 4096 \
  --max_iterations 1500 \
  --headless
```

## 5. Visualize Trained Policy

Find recent checkpoints:

```bash
find logs/rsl_rl/allegro_inhand_rotate -name "model_*.pt" -printf "%T@ %p\n" \
  | sort -n \
  | tail -5
```

Play a checkpoint with IsaacLab's existing RSL-RL play script:

```bash
./isaaclab.sh \
  -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Inhand-Rotate-Allegro-v0 \
  --num_envs 16 \
  --checkpoint <CHECKPOINT_PATH>
```

## 6. Details

- [Design philosophy](docs/design_philosophy.md)
- [Cache mechanism](docs/grasp_cache_mechanism.md)
- [Cache score guide](docs/cache_score_guide.md)

## 7. Reference

- [Sharpa RL Lab](https://github.com/sharpa-robotics/sharpa-rl-lab)
