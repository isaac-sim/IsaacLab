conda deactivate

./isaaclab.sh -p scripts/setup/download_assets.py --categories Props Robots Environments Materials Controllers ActuatorNets Policies Mimic

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
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 128 \
    --local
```

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-H1-v0 \
    --num_envs 128 \
    --local
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
    --headless \
    --num_envs 4096 \
    --resume \
    --load_run 2026-01-21_14-09-41 \
    --checkpoint model_50.pt
```

<!-- Play -->
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 128 \
    --checkpoint logs/rsl_rl/unitree_go2_flat/2026-01-21_14-38-05/model_299.pt
```

<!-- Video -->
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/velocity_flat_unitree_go2/*/model_200.pt \
    --video \
    --video_length 1000 \
    --video_interval 1
```

<!-- Video 2 -->
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 1 \
    --load_run 2025-11-29_11-15-51 \
    --checkpoint model_350.pt \
    --video \
    --video_length 500
```

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 1 \
    --load_run 2025-11-29_11-15-51 \
    --checkpoint model_350.pt \
    --video \
    --video_length 1000 \
    --video_interval 2
```

