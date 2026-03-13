# Test Configuration Generation Guide

This document explains how to generate test configurations for the Pink IK controller tests used in `test_pink_ik.py`.

## File Structure

Test configurations are JSON files with the following structure:

```json
{
    "tolerances": {
        "position": ...,
        "pd_position": ...,
        "rotation": ...,
        "check_errors": true
    },
    "allowed_steps_to_settle": ...,
    "tests": {
        "test_name": {
            "left_hand_pose": [...],
            "right_hand_pose": [...],
            "allowed_steps_per_motion": ...,
            "repeat": ...
        }
    }
}
```

## Parameters

### Tolerances
- **position**: Maximum position error in meters
- **pd_position**: Maximum PD controller error in meters
- **rotation**: Maximum rotation error in radians
- **check_errors**: Whether to verify errors (should be `true`)

### Test Parameters
- **allowed_steps_to_settle**: Initial settling steps (typically 100)
- **allowed_steps_per_motion**: Steps per motion phase
- **repeat**: Number of test repetitions
- **requires_waist_bending**: Whether the test requires waist bending (boolean)

## Coordinate System

### Robot Reset Pose
From `g1_locomanipulation_robot_cfg.py`:
- **Base position**: (0, 0, 0.75) - 75cm above ground
- **Base orientation**: 90° rotation around X-axis (facing forward)
- **Joint positions**: Standing pose with slight knee bend

### EEF Pose Format
Each pose: `[x, y, z, qw, qx, qy, qz]`
- **Position**: Cartesian coordinates relative to robot base frame
- **Orientation**: Quaternion relative to the world. Typically you want this to start in the same orientation as robot base. (e.g. if robot base is reset to (0.7071, 0.0, 0.0, 0.7071), hand pose should be the same)

**Note**: The system automatically compensates for hand rotational offsets, so specify orientations relative to the robot's reset orientation.

## Creating Configurations

### Step 1: Choose Robot Type
- `pink_ik_g1_test_configs.json` for G1 robot
- `pink_ik_gr1_test_configs.json` for GR1 robot

### Step 2: Define Tolerances
```json
"tolerances": {
    "position": 0.003,
    "pd_position": 0.001,
    "rotation": 0.017,
    "check_errors": true
}
```

### Step 3: Create Test Movements
Common test types:
- **stay_still**: Same pose repeated
- **horizontal_movement**: Side-to-side movement
- **vertical_movement**: Up-and-down movement
- **rotation_movements**: Hand orientation changes

### Step 4: Specify Hand Poses
```json
"horizontal_movement": {
    "left_hand_pose": [
        [-0.18, 0.1, 0.8, 0.7071, 0.0, 0.0, 0.7071],
        [-0.28, 0.1, 0.8, 0.7071, 0.0, 0.0, 0.7071]
    ],
    "right_hand_pose": [
        [0.18, 0.1, 0.8, 0.7071, 0.0, 0.0, 0.7071],
        [0.28, 0.1, 0.8, 0.7071, 0.0, 0.0, 0.7071]
    ],
    "allowed_steps_per_motion": 100,
    "repeat": 2,
    "requires_waist_bending": false
}
```

## Pose Guidelines

### Orientation Examples
- **Default**: `[0.7071, 0.0, 0.0, 0.7071]` (90° around X-axis)
- **Z-rotation**: `[0.5, 0.0, 0.0, 0.866]` (60° around Z)
- **Y-rotation**: `[0.866, 0.0, 0.5, 0.0]` (60° around Y)

## Testing Process

1. Robot starts in reset pose and settles
2. Moves through each pose in sequence
3. Errors computed and verified against tolerances
4. Sequence repeats specified number of times

### Waist Bending Logic
Tests marked with `"requires_waist_bending": true` will only run if waist joints are enabled in the environment configuration. The test system automatically detects waist capability by checking if waist joints (`waist_yaw_joint`, `waist_pitch_joint`, `waist_roll_joint`) are included in the `pink_controlled_joint_names` list.

## Troubleshooting

- **Can't reach target**: Check if within safe workspace
- **High errors**: Increase tolerances or adjust poses
- **Test failures**: Increase `allowed_steps_per_motion`
