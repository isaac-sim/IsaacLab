# Isaac Lab Colab notebooks

These notebooks are self-contained introductions to Isaac Lab 3.0. They clone the
Isaac Lab `develop` branch into the Colab VM and install its pinned dependencies;
they do not modify the checkout from which the notebook was opened.

| Notebook | What it covers |
|---|---|
| [`explore.ipynb`](explore.ipynb) | Scenes and presets, the robot asset library, a pretrained policy, cables, cloth, and soft bodies, solver coupling, and batched cameras |
| [`training.ipynb`](training.ipynb) | A complete locomotion workflow: customize a maintained task, train it with live learning curves, and replay its checkpoints, all in the notebook process |

[![Open the explore notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isaac-sim/IsaacLab/blob/develop/notebooks/explore.ipynb)
[![Open the training notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isaac-sim/IsaacLab/blob/develop/notebooks/training.ipynb)

## Before running

1. Open a notebook in Google Colab.
2. Select **Runtime > Change runtime type > GPU**. An L4 or A100 runtime is
   recommended; free-tier hardware and memory availability vary.
3. Run the installation cell, restart the session once, and run the remaining cells in order.

The notebooks run headless: every example records a video from a camera sensor and
plays it beneath its cell. Physics runs on Newton with MuJoCo Warp (VBD for
deformables), and cameras render with OV RTX. A setup cell compiles the OV RTX
shaders once, which takes a couple of minutes on a fresh runtime; later videos
start in seconds. Pretrained checkpoints are downloaded on demand from the Isaac Lab
asset server.

Colab forms (dropdowns and hidden code cells) only render in Colab. In a local
Jupyter session they appear as ordinary code; edit the assigned values directly.

For a reproducible workshop, change `ISAACLAB_REF` in the installation cell of each
notebook from `develop` to the exact Isaac Lab 3.0 release tag used by the class.
