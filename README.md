<img width="5152" height="2528" alt="image" src="https://github.com/user-attachments/assets/d65a0dad-00ac-4ab2-8849-91b8ba8e604a" />

# Record dataset

```
cd ~/Stanley_ws/IsaacLab
```

```
conda activate env_isaaclab
```

```
cd IsaacLab/
./isaaclab.sh --install
```

```
./isaaclab.sh -p scripts/tools/record_demos_openarm.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0     --dataset_file logs/demos/pickup.hdf5     --enable_cameras --num_demos 1 --teleop_device keyboard
```

```
=== OpenArm Dual-Arm Recording ===
Key	                  Action
W / S	                EE forward / backward (+X / -X)
A / D	                EE left / right (+Y / -Y)
PgUp / PgDn	            EE up / down (+Z / -Z)
↑ / ↓	                pitch ±
← / →	                yaw ±
[ / ]	                roll ±
K	                    gripper toggle
TAB	                    switch arm (left ↔ right)
N	                    save episode
R	                    reset/discard

```


# Replay dataset

```
cd ~/Stanley_ws/IsaacLab
```

```
conda activate env_isaaclab
```

```
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --dataset_file logs/demos/pickup_test.hdf5 \
    --enable_cameras
```

# Environment Setup

If you want to change environment in IsaacSim, please refer to the following setting

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_joint_pos_env_cfg.py
```

### Table / Workspace (Pad)

PAD_HEIGHT = 0.13          # height of the gray platform (m)

CUBE_Z = PAD_HEIGHT + 0.02 # cube resting height = pad top + half-block

### Robot Base Height

```
self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)  # adjust Z to raise/lower
```

### Cube Size and Color

```
for i, (name, pos, color) in enumerate([
    ("cube_1", [0.2,  0.08,  CUBE_Z], "blue"),   # change "blue" → any color name
    ("cube_2", [0.55, 0.05,  CUBE_Z], "red"),     # the pickup cube
    ("cube_3", [0.60, -0.10, CUBE_Z], "green")
]):
    spawn=UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/{color}_block.usd",  # USD determines size too
    )
```

### Cube Randomization Range (each episode)

```
randomize_cube_2 = EventTerm(
    func=mdp_core.reset_root_state_uniform,
    params={
        "pose_range": {
            "x": (-0.38, -0.28),  # offset from default x=0.55 → actual x:[0.17, 0.27]
            "y": (-0.02,  0.12),  # offset from default y=0.05 → actual y:[0.03, 0.17]
        },
        ...
    },
)
```

# Isaac Lab Mimic

```
cd ~/Stanley_ws/IsaacLab
```

```
conda activate env_isaaclab
```

Record source demo (keyboard teleoperation)

```
./isaaclab.sh -p scripts/tools/record_demos_openarm.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --dataset_file logs/demos/pickup.hdf5 \
    --enable_cameras --num_demos 10 --teleop_device keyboard
```

Annotate with subtask signals (auto-mode uses get_subtask_term_signals)

```
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
    --input_file logs/demos/pickup.hdf5 \
    --output_file logs/demos/pickup_annotated.hdf5 --auto --enable_cameras
```

Generate augmented dataset

```
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Mimic-v0 \
    --input_file logs/demos/pickup_annotated.hdf5 \
    --output_file logs/demos/pickup_generated.hdf5 \
    --generation_num_trials 50 --num_envs 4 --enable_cameras
```

## Generate augmented dataset(Cosmos-transfer)

```
git clone https://github.com/nvidia-cosmos/cosmos-transfer1.git
cd cosmos-transfer1
git checkout e4055e39ee9c53165e85275bdab84ed20909714a   # tested commit

conda create -n cosmos python=3.10 -y
conda activate cosmos
pip install -e .
```

```
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py     --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-Cosmos-Mimic-v0     --input_file logs/demos/pickup_source_annotated.hdf5     --output_file logs/demos/pickup_source_generated_10_cosmos.hdf5     --generation_num_trials 10 --num_envs 4 --enable_cameras     --rendering_mode performance
```

```
 python scripts/tools/hdf5_to_mp4.py \
    --input_file logs/demos/pickup_source_generated_10_cosmos.hdf5 \
    --output_dir logs/demos/pickup_cosmos_mp4 \
    --input_keys front_cam wrist_cam body_cam \
                 front_cam_segmentation front_cam_normals front_cam_depth \
                 wrist_cam_depth \
    --video_height 704 --video_width 1280 --framerate 30
```

```
cd ~/Stanley_ws/cosmos-transfer1
/home/csl/Stanley_ws/IsaacLab/scripts/cosmos/run_cosmos_inference.sh
```

# Convert HDF5 to LeRobot format 

```
cd ~/Stanley_ws/IsaacLab
```

```
conda activate env_isaaclab
```

```
python -u scripts/tools/convert_hdf5_to_lerobot.py     --hdf5 logs/demos/pickup_source.hdf5     --output ~/Stanley_ws/IsaacLab/datasets/ethanCSL/openarm_visuomotor     --task "Pick up the red cube."     --fps 30 --cameras front_cam wrist_cam body_cam
```

# Train in LeRobot format

```
cd ~/CSL/lerobot/
```

```
conda activate lerobot
```

```
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/openarm_visuomotor   --batch_size=16   --steps=40000   --output_dir=outputs/train/openarm_visuomotor   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/openarm_visuomotor  --wandb.enable=false   --rename_map='{
    "observation.images.front_cam": "observation.images.camera1",
    "observation.images.body_cam":   "observation.images.camera2",
    "observation.images.wrist_cam":  "observation.images.camera3"
  }'   --dataset.video_backend=pyav

```

# Deploy in Isaac Sim

Launch SmolVLA Policy Server

```
conda activate lerobot
```

```
python ~/Stanley_ws/IsaacLab/scripts/imitation_learning/lerobot/smolvla_server.py \
    --checkpoint ethanCSL/openarm_visuomotor_augmented_dataset_1000 \
    --task "Pick up the red cube." \
    --port 5556
```

Run Isaac Lab Eval

```
conda activate env_isaaclab
```

```
cd ~/Stanley_ws/IsaacLab
./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_smolvla.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --num_rollouts 5 \
    --horizon 300 \
    --enable_cameras
```

# Run in Real-world

```
cd ~/openarm_can/setup
```

```
sudo ./my_arm 
```

## OpenARM Motors Check

```
cd lerobot_openarm
```

```
uv sync
source .venv/bin/activate
```

You can change joint number and arm to different testing 

```
python safe_probe.py --side left --joint 1 --step 0.1 --max-kp 150 --skip-ctrl-mode
```
## OpenARM Teleoperation Mirror Test

```
./isaaclab.sh -p scripts/tools/record_demos_openarm.py \
    --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \
    --dataset_file logs/demos/pickup.hdf5 \
    --enable_cameras --num_demos 10 --teleop_device keyboard \
    --mirror_udp_port 9999
```

```
python mirror_bridge.py --calibration calibration.json --udp-port 9999     --right-port can2 --left-port can3     --model-path model/openarm_description_leader.urdf     --max-joint-speed 0.5
```




