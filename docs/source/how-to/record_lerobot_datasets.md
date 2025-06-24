# Recording Datasets in LeRobot Format

This guide explains how to record demonstration datasets in LeRobot format instead of the default HDF5 format. LeRobot format is designed for training Vision-Language-Action (VLA) models and is compatible with the [🤗 LeRobot](https://github.com/huggingface/lerobot) ecosystem.

## Why Use LeRobot Format?

The LeRobot format offers several advantages over traditional HDF5 storage:

- **Standardized Format**: Compatible with the growing LeRobot ecosystem
- **Efficient Storage**: Uses MP4 videos for camera observations and Parquet files for tabular data
- **Rich Metadata**: Includes task descriptions, episode information, and standardized feature naming
- **Easy Sharing**: Can be uploaded to HuggingFace Hub for community sharing
- **Training Ready**: Direct compatibility with LeRobot training pipelines

## Installation

First, install the required dependencies for LeRobot format:

```bash
pip install datasets opencv-python imageio[ffmpeg]
```

## Basic Usage

### Using the LeRobot Dataset Handler

```python
from isaaclab.utils.datasets import LeRobotDatasetFileHandler, EpisodeData

# Create a new LeRobot dataset
handler = LeRobotDatasetFileHandler()
handler.create("./datasets/my_dataset.lerobot", env_name="Isaac-Stack-Cube-Franka-IK-Rel-v0")

# Record episode data (example)
episode = EpisodeData()
# ... populate episode with observations and actions ...

# Write episode to dataset
handler.write_episode(episode)
handler.flush()
handler.close()
```

### Modifying Recording Scripts

To use LeRobot format instead of HDF5 in your recording scripts, modify the recorder manager configuration:

```python
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, DatasetExportMode
from isaaclab.utils.datasets import LeRobotDatasetFileHandler

# Configure recorder to use LeRobot format
recorder_cfg = RecorderManagerBaseCfg()
recorder_cfg.dataset_file_handler_class_type = LeRobotDatasetFileHandler
recorder_cfg.dataset_export_dir_path = "./datasets"
recorder_cfg.dataset_filename = "my_lerobot_dataset"
recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
```

## Dataset Structure

LeRobot datasets follow a standardized structure:

```
dataset.lerobot/
├── dataset_info.json          # HuggingFace dataset metadata
├── state.json                 # Dataset state information
├── data/                      # Parquet files with episode data
│   ├── train-00000-of-00001.parquet
│   └── ...
├── videos/                    # Video files for camera observations
│   ├── episode_000000/
│   │   ├── front.mp4
│   │   ├── wrist.mp4
│   │   └── ...
│   └── episode_000001/
│       └── ...
└── meta/                      # Additional metadata
    └── info.json              # Isaac Lab specific metadata
```

## Feature Naming Conventions

LeRobot uses standardized naming conventions for observations:

- **Camera observations**: `observation.images.{camera_position}`
  - Examples: `observation.images.front`, `observation.images.wrist`, `observation.images.top`
- **Robot state**: `observation.state`
- **Actions**: `action`
- **Episode metadata**: `episode_index`, `frame_index`, `timestamp`, `task`

## Using the Recording Script

The easiest way to record demonstrations in LeRobot format is using the built-in recording script:

```bash
# Record in LeRobot format with SpaceMouse
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device spacemouse \
    --dataset_file ./datasets/my_dataset.lerobot \
    --lerobot_format \
    --num_demos 10

# Record in LeRobot format with hand tracking
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device handtracking \
    --lerobot_format \
    --num_demos 5

# Record in LeRobot format with keyboard
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
    --teleop_device keyboard \
    --lerobot_format \
    --num_demos 3
```

The script automatically handles:
- Dependency checking for LeRobot format
- File extension conversion (.hdf5 → .lerobot)
- Recorder manager configuration
- All teleoperation devices (keyboard, spacemouse, handtracking)

## Programmatic Usage

For custom integration into your own scripts:

```python
import gymnasium as gym
from isaaclab.utils.datasets import LeRobotDatasetFileHandler
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

def setup_lerobot_recording(env_cfg):
    """Configure environment for LeRobot recording."""
    
    # Configure recorder for LeRobot format
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = "./datasets"
    env_cfg.recorders.dataset_filename = "my_lerobot_dataset"
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = LeRobotDatasetFileHandler
    
    return env_cfg

# Use in your environment
env_cfg = setup_lerobot_recording(env_cfg)
env = gym.make("Isaac-Stack-Cube-Franka-IK-Rel-v0", cfg=env_cfg)
```

## Uploading to HuggingFace Hub

Once you've recorded your dataset, you can share it with the community:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login

# Upload dataset
huggingface-cli upload <your-username>/<dataset-name> ./datasets/my_dataset.lerobot
```

## Training with LeRobot

After uploading your dataset, you can train models using the LeRobot pipeline:

```bash
# Install LeRobot
pip install lerobot

# Train a model
python lerobot/scripts/train.py \
    --dataset-repo-id <your-username>/<dataset-name> \
    --policy-name diffusion \
    --env-name <your-env-name>
```

## Comparison with HDF5 Format

| Feature | HDF5 Format | LeRobot Format |
|---------|-------------|----------------|
| Storage | Single HDF5 file | Directory with Parquet + MP4 |
| Video compression | Raw arrays | Efficient MP4 encoding |
| Metadata | Basic | Rich metadata with task descriptions |
| Sharing | Manual file transfer | HuggingFace Hub integration |
| Training compatibility | Isaac Lab only | LeRobot ecosystem |
| File size | Larger | Smaller (compressed videos) |

## Best Practices

1. **Use descriptive task names**: Provide clear task descriptions that will help with training
2. **Consistent camera naming**: Follow LeRobot conventions for camera positions
3. **Quality data**: Ensure clean demonstrations with clear success/failure labels
4. **Metadata**: Include rich metadata about the task, environment, and recording conditions
5. **Version control**: Use semantic versioning for dataset uploads

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure LeRobot dependencies are installed
   ```bash
   pip install datasets opencv-python imageio[ffmpeg]
   ```

2. **Video encoding issues**: Make sure FFmpeg is properly installed
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # On macOS
   brew install ffmpeg
   ```

3. **Large file sizes**: Videos are compressed by default, but you can adjust quality settings in the handler

### Getting Help

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)

---

The LeRobot format enables seamless integration with the broader robotics AI ecosystem while maintaining compatibility with Isaac Lab's recording infrastructure. This makes it easier to share datasets, collaborate with the community, and leverage state-of-the-art VLA models for your robotics applications. 