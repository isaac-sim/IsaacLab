Training & Deploying COMPASS Navigation Policy with Real2Sim NuRec
====================================================================

COMPASS (Cross-embodiment Mobility Policy via Residual RL and Skill Synthesis) trains and
deploys cross-embodiment navigation policies on NuRec Real2Sim assets — photoreal
Gaussian-splat reconstructions of real spaces — in Isaac Lab.

This tutorial now lives in the **COMPASS handbook**, next to the code it documents, where it
stays in sync with the NuRec support branch.

Where to find it
----------------

- **Repository:** `COMPASS on GitHub <https://github.com/NVlabs/COMPASS>`_
- **Branch:** ``samc/support_nurec_assets_isaaclab_3.0`` (NuRec Real2Sim support)
- **Guide:** `docs/handbook/workflows/nurec_real2sim.md <https://github.com/NVlabs/COMPASS/blob/samc/support_nurec_assets_isaaclab_3.0/docs/handbook/workflows/nurec_real2sim.md>`_ in the COMPASS repository.

.. code-block:: bash

    git clone https://github.com/NVlabs/COMPASS.git
    cd COMPASS
    git checkout samc/support_nurec_assets_isaaclab_3.0
    # Open docs/handbook/workflows/nurec_real2sim.md

The guide covers Isaac Sim / Isaac Lab installation, COMPASS setup, downloading and
registering NuRec Real2Sim scenes, training a residual RL specialist, evaluation,
ONNX / TensorRT export, and ROS2 / sim-to-real deployment.
