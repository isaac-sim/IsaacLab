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
