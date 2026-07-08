# Allegro Grasp Cache Mechanism

The Allegro rotate task uses a two-stage workflow:

```text
grasp-cache generation -> rotate-policy training
```

The grasp-cache stage does not train a policy. It samples hand/object states around the configured ready pose, holds them with zero actions, filters stable states, and saves successful reset states to `.npy`.

## What Zero Action Means

In this environment, action is interpreted as a change to the current joint target:

```text
target = previous_target + action_scale * action
```

So:

```text
action = 0
```

means:

```text
keep the current joint target
```

It does not mean a limp hand. The position controller can still apply force to hold the sampled pose.

## What The Cache Saves

One Allegro cache row stores:

```text
hand_dof_pos(16) + object_pos(3) + object_rot(4) = 23 floats
```

The cache does not save actions, rewards, contact forces, velocities, or trajectories.

## Cache Command

```bash
./isaaclab.sh \
  -p source/isaaclab_tasks/isaaclab_tasks/direct/allegro_rotate/tools/allegro_gen_grasp.py \
  --task Isaac-Inhand-Rotate-Grasp-Allegro-v0 \
  --num_envs 4096 \
  --max_cache_rows 50000 \
  --output cache/allegro_grasp_linspace \
  --headless
```

Expected output:

```text
cache/allegro_grasp_linspace_0.8-0.8-1.npy
```

## Visualization

```bash
./isaaclab.sh \
  -p source/isaaclab_tasks/isaaclab_tasks/direct/allegro_rotate/tools/allegro_viz_cache.py \
  --cache cache/allegro_grasp_linspace_0.8-0.8-1.npy \
  --num_envs 1 \
  --viz kit \
  --real-time
```

Train only after cache visualization shows a stable supported grasp.
