# Installation

```bash
# Create and activate a uv virtual environment
cd ./projects/IsaacLab_release_3_0
uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate

uv pip install "isaacsim[all,extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
./isaaclab.sh --install
```

# Training command

```bash
source env_isaaclab/bin/activate
cd ./projects/IsaacLab_release_3_0

./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --headless physics=newton_mjwarp
```

[INFO] Logging experiment in directory: ./ReseachOS/IsaacLab_release_3_0/logs/rsl_rl/franka_deformable

# Eval
```bash
source env_isaaclab/bin/activate
cd ./projects/IsaacLab_release_3_0

./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 \
  --headless \
  --video --video_length 400 \
  --num_envs 32 \
  --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-06-11_15-09-40/model_999.pt
```

# Export reward and success-rate curves from TensorBoard
```bash
source env_isaaclab/bin/activate
cd ./projects/IsaacLab_release_3_0

./isaaclab.sh -p scripts/reinforcement_learning/export_training_curves.py \
  --checkpoint logs/rsl_rl/dexsuite_kuka_allegro/2026-06-11_15-09-40/model_999.pt
```