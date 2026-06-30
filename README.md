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

If you want to change environment in IsaacSim, please refer to 

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_joint_pos_env_cfg.py
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

