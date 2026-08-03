# Motion files

The motion files are in NumPy-file format that contains data from the skeleton DOFs and bodies that perform the motion.

The data (accessed by key) is described in the following table, where:

* `N` is the number of motion frames recorded
* `D` is the number of skeleton DOFs
* `B` is the number of skeleton bodies

| Key | Dtype | Shape | Description |
| --- | ---- | ----- | ----------- |
| `fps` | int64 | () | FPS at which motion was sampled |
| `dof_names` | unicode string | (D,) | Skeleton DOF names |
| `body_names` | unicode string | (B,) | Skeleton body names |
| `dof_positions` | float32 | (N, D) | Skeleton DOF positions |
| `dof_velocities` | float32 | (N, D) | Skeleton DOF velocities |
| `body_positions` | float32 | (N, B, 3) | Skeleton body positions |
| `body_rotations` | float32 | (N, B, 4) | Skeleton body rotations (as `wxyz` quaternion) |
| `body_linear_velocities` | float32 | (N, B, 3) | Skeleton body linear velocities |
| `body_angular_velocities` | float32 | (N, B, 3) | Skeleton body angular velocities |

## Motion visualization

The `motion_viewer.py` file allows to visualize the skeleton motion recorded in a motion file.

Open an terminal in the `motions` folder and run the following command.

```bash
python motion_viewer.py --file MOTION_FILE_NAME.npz
```

See `python motion_viewer.py --help` for available arguments.
