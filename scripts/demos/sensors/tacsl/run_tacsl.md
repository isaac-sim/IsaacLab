
This is an implementation of TacSL integrated with Isaac Lab, which demonstrates how to properly configure and use tactile sensors to obtain realistic sensor outputs including tactile RGB images, force fields, and other relevant tactile measurements.


---

## Setup

### Prerequisites
Please follow the [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to install IsaacLab first.

### Additional Dependencies
Install the required additional dependencies:
```bash
conda activate env_isaaclab
pip install opencv-python==4.11.0 trimesh==4.5.1
```

### Assets
Download the Gelsight assets from [here](https://drive.google.com/drive/folders/1SyLF8AI0W8I9r9zCEKVO4bB804THFMGD?usp=drive_link) and place them in the `./assets/` folder.

---
## Usage

### Verify Installation
To verify your IsaacLab installation, you can train a robot policy:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

### Run TacSL Demo
To run the tactile sensor demonstration with shape sensing:
```bash
cd scripts/demos/sensors/tacsl
python tacsl_example.py --enable_cameras --indenter nut --num_envs 8 --use_tactile_taxim --use_tactile_ff --save_viz
```

### Available Options
For a complete list of available command-line arguments and options:
```bash
cd scripts/demos/sensors/tacsl
python tacsl_example.py -h
```

## What to Expect

The demo showcases tactile sensor simulation with various object interactions. You'll see:
- Real-time tactile RGB image generation as objects contact the sensor surface
- Force field visualizations showing contact forces and pressure distributions
- Shape sensing capabilities for object recognition and analysis
- Configurable sensor parameters and object interactions

The visualization outputs can be saved for further analysis and research purposes.
