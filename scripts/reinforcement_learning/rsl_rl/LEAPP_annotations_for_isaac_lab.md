# LEAPP Export for Isaac Lab

Export RSL-RL reinforcement learning pipelines as portable processing graphs using [LEAPP](https://gitlab-master.nvidia.com/Isaac/leapp).

## Exported Artifacts

| File | Description |
|------|-------------|
| `<taskname>.onnx` | Policy network (ONNX) |
| `<task_name>.yaml` | Pipeline configuration and metadata |
| `<task_name>.png` | Visualization of the processing graph |

The YAML file includes semantic metadata (joint names, units, etc.) extracted from IO descriptors. For details on the YAML format, see the [LEAPP documentation](https://gitlab-master.nvidia.com/Isaac/leapp/-/blob/main/docs/0_getting_started.md).

## Usage

### 1. Install LEAPP

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/Isaac/leapp.git
cd leapp
git checkout develop
pip install -e .
```

### 2. Export a Policy

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/export.py \
    --task Isaac-Reach-Franka-v0 \
    --use_pretrained_checkpoint \
    --headless
```

> **Note:** Export runs with a single environment instance.

### 3. View Results

Artifacts are saved to `./<task_name>/`.



sample exported `Isaac-Reach-Franka-v0.yaml`:

```yaml
models:
  Isaac-Reach-Franka-v0:
    inputs:
    - name: joint_pos
      dtype: float32
      shape: [1, 9]
      type: tensor
    - name: joint_vel
      dtype: float32
      shape: [1, 9]
      type: tensor
    - name: ee_pose
      dtype: float32
      shape: [1, 7]
      type: tensor
    - name: last_actions
      dtype: float32
      shape: [1, 7]
      type: tensor
    outputs:
    - name: arm_action
      dtype: float32
      shape: [1, 7]
      type: tensor
    - name: last_action
      dtype: float32
      shape: [1, 7]
      type: tensor
    - name: arm_action_kp_gains
      dtype: float32
      shape: [1, 7]
      type: tensor
    - name: arm_action_kd_gains
      dtype: float32
      shape: [1, 7]
      type: tensor
    parameters:
      model_path: Isaac-Reach-Franka-v0.onnx
      md5sum: 38ee55fa7828b5068b86024206bd5ddb
      sha256sum: c605a7076fde5c0d03a36f548d458d24bd543df67aac7675d463d29f870a7eb3
      device: cuda
      backend: onnx

pipeline:
  data_flow: {}
  feedback_flow:
    Isaac-Reach-Franka-v0/last_action: [Isaac-Reach-Franka-v0/last_actions]
  inputs:
    Isaac-Reach-Franka-v0: [joint_pos, joint_vel, ee_pose]
  outputs:
    Isaac-Reach-Franka-v0: [arm_action, arm_action_kp_gains, arm_action_kd_gains]

system information:
  cuda version: '12.8'
  leapp version: 0.3.0
  os: Linux
  python version: 3.11.14
  torch version: 2.7.0+cu128

semantic:
  actions:
  - joint_names:
    - panda_joint1
    - panda_joint2
    - panda_joint3
    - panda_joint4
    - panda_joint5
    - panda_joint6
    - panda_joint7
    leapp_mapping:
    - arm_action
    name: joint_position_action
  observations:
  - joint_names:
    - panda_joint1
    - panda_joint2
    - panda_joint3
    - panda_joint4
    - panda_joint5
    - panda_joint6
    - panda_joint7
    - panda_finger_joint1
    - panda_finger_joint2
    leapp_mapping:
    - joint_pos
    name: joint_pos_rel
    units: rad
  - joint_names:
    - panda_joint1
    - panda_joint2
    - panda_joint3
    - panda_joint4
    - panda_joint5
    - panda_joint6
    - panda_joint7
    - panda_finger_joint1
    - panda_finger_joint2
    leapp_mapping:
    - joint_vel
    name: joint_vel_rel
    units: rad/s
  - leapp_mapping:
    - ee_pose
    name: generated_commands
  - leapp_mapping:
    - last_actions
    name: last_action
  scene:
    decimation: 2
    dt: 0.03333333333333333
    physics_dt: 0.016666666666666666

```
