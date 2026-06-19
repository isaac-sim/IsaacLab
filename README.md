# Record dataset

```
 ./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Stack-Cube-OpenArm-IK-Abs-Visuomotor-v0 --dataset_file logs/demos/visuomotor.hdf5 --enable_cameras  --num_demos 1
```

# Replay dataset

```
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-Stack-Cube-OpenArm-IK-Abs-Visuomotor-v0 \
    --dataset_file logs/demos/visuomotor.hdf5 \
    --enable_cameras
```

# Convert HDF5 to LeRobot format 

```
conda run -n lerobot python scripts/tools/convert_hdf5_to_lerobot.py --hdf5 logs/demos/visuomotor.hdf5 --output ~/Stanley_ws/IsaacLab/datasets/openarm_visuomotor --task "OpenArm" --fps 20 --cameras front_cam wrist_cam
```

# Train in LeRobot format

```
 lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=ethanCSL/openarm_visuomotor \
    --batch_size=16 \
    --steps=20000 \
    --output_dir=outputs/train/openarm_visuomotor \
    --job_name=my_smolvla_training \
    --policy.device=cuda \
    --policy.repo_id=ethanCSL/openarm_visuomotor \
    --wandb.enable=false \
    --rename_map='{"observation.images.front_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2"}' \
    --policy.empty_cameras=1 \
    --dataset.video_backend=pyav
```
