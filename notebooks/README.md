# Isaac Lab Colab notebooks

These notebooks are self-contained introductions to Isaac Lab 3.0. They install Isaac Lab from the
`develop` branch and its dependencies in the Colab VM;
they do not modify the checkout from which the notebook was opened.

| Notebook | What it covers |
|---|---|
| [`explore.ipynb`](explore.ipynb) | Scenes and presets, the robot asset library, a pretrained policy, cables, cloth, and soft bodies, solver coupling, and batched cameras |
| [`training.ipynb`](training.ipynb) | A complete locomotion workflow: customize a maintained task, train it with live learning curves, and replay its checkpoints, all in the notebook process |

[![Open the explore notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isaac-sim/IsaacLab/blob/develop/notebooks/explore.ipynb)
[![Open the training notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isaac-sim/IsaacLab/blob/develop/notebooks/training.ipynb)

## Before running

1. Open a notebook in Google Colab.
2. Open **Runtime → Change runtime type**.
3. Set **Runtime Version** to **2026.07**, which includes **Python 3.12.13**.
4. Select your **GPU** accelerator and reconnect. An L4 or A100 runtime is
   recommended; free-tier hardware and memory availability vary.
5. Run the installation cell, restart the session once, and run the remaining cells in order.

Python 3.12 and an NVIDIA CUDA GPU are required. Warp supports CPU and CUDA execution,
but this notebook workflow requires CUDA; TPU runtimes are not supported. See the [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html)
and [Warp requirements](https://github.com/NVIDIA/warp#installing).

The notebooks run headless: every example records a video from a camera sensor and
plays it beneath its cell. Physics runs on Newton with MuJoCo Warp (VBD for
deformables), and cameras use the Newton Warp renderer. This avoids OV RTX's
requirement for RTX-capable hardware, which GPUs such as the A100 do not meet.
A setup cell warms up physics and rendering; the first run compiles Warp kernels
and later runs reuse the cache. Full hosted-Colab execution has not yet been validated.
Pretrained checkpoints are downloaded on demand from the Isaac Lab
asset server.

With a local runtime, you can install the `ov` extra and set `RENDERER = "ovrtx"`
in the backend-selection cell. This requires compatible RTX hardware and
[OV RTX drivers](https://github.com/NVIDIA-Omniverse/ovrtx/blob/main/docs/driver_requirements.rst).
The warmup cell can keep using Newton.

Colab forms (dropdowns and hidden code cells) only render in Colab. In a local
Jupyter session they appear as ordinary code; edit the assigned values directly.

For a reproducible workshop, change `@develop` in the installation command of each
notebook to `@<release-tag>` for the exact Isaac Lab 3.0 release tag used by the class.
