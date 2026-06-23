<img width="5152" height="1216" alt="image" src="https://github.com/user-attachments/assets/e79fc2ab-05ae-4731-9c1c-a3385cf7facc" />

# Conda Environment

```
conda activate env_isaaclab
```

# Record dataset

```
./isaaclab.sh -p scripts/tools/record_demos_openarm.py --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0  --dataset_file logs/demos/visuomotor.hdf5 --enable_cameras  --num_demos 1 --teleop_device  keyboard
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
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-Reach-RedCube-OpenArm-IK-Abs-v0 \
    --dataset_file logs/demos/visuomotor.hdf5 \
    --enable_cameras
```

# Environment Setup

If you want to change environment in IsaacSim, please refer to 

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/config/franka/stack_joint_pos_env_cfg.py
```

# Isaac Lab Mimic

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
conda run -n lerobot python -u scripts/tools/convert_hdf5_to_lerobot.py     --hdf5 logs/demos/pickup.hdf5     --output ~/Stanley_ws/IsaacLab/datasets/ethanCSL/openarm_visuomotor     --task "Pick up the red cube."     --fps 30     --cameras front_cam wrist_cam body_cam
```

# Train in LeRobot format

```
cd ~/CSL/lerobot/
```

```
 lerobot-train   --policy.path=lerobot/smolvla_base   --dataset.repo_id=ethanCSL/openarm_visuomotor   --batch_size=16   --steps=40000   --output_dir=outputs/train/openarm_visuomotor   --job_name=my_smolvla_training   --policy.device=cuda   --policy.repo_id=ethanCSL/openarm_visuomotor  --wandb.enable=false   --rename_map='{
    "observation.images.front_cam": "observation.images.camera1",
    "observation.images.body_cam":   "observation.images.camera2",
    "observation.images.wrist_cam":  "observation.images.camera3"
  }'   --dataset.video_backend=pyav

```
