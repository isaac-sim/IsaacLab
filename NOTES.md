conda deactivate

<!-- Running Local Training Headless -->
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --headless \
    --num_envs 128
```

<!-- Running Local Training -->
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-H1-v0 \
    --num_envs 496 \
    --offline
```

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 128 \
    --offline
```

<!-- Resume Training from latest checkpoint -->

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 4096 \
    --resume
```

<!-- Resuming from specific checkpoint -->
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 4096 \
    --resume \
    --load_run 2026-01-21_14-09-41 \
    --checkpoint model_50.pt
```

<!-- Play -->
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-H1-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/h1_flat/2026-01-27_14-58-33/model_800.pt \
    --offline
```

<!-- Video -->
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-H1-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/h1_flat/2026-01-27_14-58-33/model_800.pt \
    --video \
    --video_length 1000 \
    --offline
```