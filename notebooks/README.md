# Isaac Lab Colab notebooks

These notebooks are self-contained introductions to Isaac Lab 3.0. They clone the
Isaac Lab `develop` branch into the Colab VM and create a Python 3.12 environment
with `uv`; they do not modify the checkout from which the notebook was opened.

| Notebook | What it covers |
|---|---|
| [`explore.ipynb`](explore.ipynb) | Pretrained policies, physics and renderer presets, camera observations, and deformable objects |
| [`training.ipynb`](training.ipynb) | A complete manager-based, state-observation locomotion workflow: author, register, smoke-test, train, inspect, and play |

[![Open the showcase in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isaac-sim/IsaacLab/blob/develop/notebooks/explore.ipynb)
[![Open the training flow in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isaac-sim/IsaacLab/blob/develop/notebooks/training.ipynb)

## Before running

1. Open a notebook in Google Colab.
2. Select **Runtime > Change runtime type > GPU**. An L4 or A100 runtime is
   recommended; free-tier hardware and memory availability vary.
3. Run cells in order. The first installation can take several minutes.

The notebooks use kitless backends and headless video capture because a Colab VM
cannot expose an interactive Isaac Sim viewport. Published checkpoints are fetched
on demand. Their availability depends on the selected task/backend pair and access
to the Isaac Lab asset server; each showcase cell reports a useful recovery path if
a checkpoint is unavailable.

Each demo has a compact progress display and expandable logs. Explore includes
physics and renderer pickers and records the actual camera input for the vision
demo. Training includes learning curves, early/final checkpoint playback, and a
downloadable experiment archive. Use **Show code** in Colab to inspect or edit a cell.

For a reproducible workshop, change `ISAACLAB_REF` near the top of each notebook
from `develop` to the exact Isaac Lab 3.0 release tag used by the class.
