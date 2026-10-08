.. _so101_typing_policy:

SO-101 Keyboard Typing with Isaac Lab
=====================================

The `SO-101 typing project <https://github.com/NVIDIA>`_ trains
an SO-101 follower to press letters on a fixed Logitech MX Keys keyboard with
its typing jaw, then runs the learned policy on hardware through LeRobot.
Use it to explore sim-to-real typing, play a released policy in simulation,
or calibrate and train a policy for your own robot and keyboard setup.

.. note::

   **Isaac Lab version.** This project uses a pinned, patched Isaac Lab 3.0.0-based
   runtime built from source revision
   `6f991d4becf7 <https://github.com/ooctipus/IsaacLab/commit/6f991d4becf764b151e0a1c775561ddd9c406f72>`_.
   Use the project's Docker image rather than an existing Isaac Lab installation.
   Compatibility with unmodified 3.0 GA or later versions has not been validated.
   See the `runtime provenance <https://github.com/NVIDIA>`_
   for the source patch and dependency pins.

What the project includes
-------------------------

* An Isaac Lab typing environment using Newton's MJWarp solver and configurable
  actuator models, with PPO training and evaluation through RSL-RL.
* Tools to measure the fixture and fit the simulated keyboard pose and joint-angle
  calibration from recorded real key presses.
* Released checkpoints, simulation playback, and a separate CPU deployment runner
  using LeRobot for motor control and Linux keyboard events for press/release feedback.

The policy uses a fixed A-Z key-position map without camera input; it cannot
adjust to a moved keyboard. Physical deployment requires calibration and checks
that the real robot reaches and releases the intended keys in the trained scene.

Supported platform and requirements
-----------------------------------

The documented workflow supports **Linux x86-64 only**. Windows and ARM have not
been validated for this project.

Simulation and training require an NVIDIA GPU, Docker with NVIDIA Container
Toolkit, host Python 3.10 or later, and ``curl`` or ``wget``. Build the project's
Docker image with ``./so101 build`` before running simulation commands. Allow
**at least 70 GB of available disk space** for Docker layers, caches and runs;
the built image is approximately 32.6 GB. Adjust parallel environments to fit
your GPU's memory; this also changes the number of samples per training update.

Hardware deployment additionally requires an SO-101 follower with the supported
typing jaw, a Logitech MX Keys keyboard, and ``uv`` for the separate LeRobot
environment. Hardware inference does not require Isaac Sim or a GPU. See the
`hardware preparation guide <https://github.com/NVIDIA>`_
for calibration, device access and operator procedures.

Get started in simulation
-------------------------

Once the repository is public, replace ``REPOSITORY_URL`` below with its clone
URL and build the pinned runtime:

.. code-block:: bash

   git clone REPOSITORY_URL so101-typing-task
   cd so101-typing-task
   ./so101 build

Download the released September checkpoint and generate a typing rollout;
no physical robot is needed:

.. code-block:: bash

   ./so101 download
   bundle=.artifacts/so101-keyboard-typing-benchmark/releases/calibrated-20260917
   ./so101 video --actuator anchorbench --stage transit15 \
     --checkpoint "$bundle/model_2498.pt" \
     --env-config "$bundle/params/env.yaml" --target NEWTON

Keep the checkpoint's complete ``params/`` directory for playback, evaluation
and deployment. The first simulation launch compiles Warp kernels.

Training and physical deployment
--------------------------------

Follow the repository's `training guide <https://github.com/NVIDIA>`_
for the two-letter P1A and six-letter Transit15 stages. Its demonstrated starting
budget is 500 P1A plus 2,000 Transit15 updates with 4,096 environments. Evaluate
between stages and increase the budget if needed; this is not a guaranteed minimum.
The example seeds make experiments reproducible and are not required values for success.

Before hardware execution, follow the
`scene calibration <https://github.com/NVIDIA>`_
and `alignment checks <https://github.com/NVIDIA>`_
for your robot. The released joint-angle offsets map encoder readings to model
coordinates for the original setup; they do not calibrate another robot.

The repository maintains the detailed
`simulation quickstart <https://github.com/NVIDIA>`_,
`fixture setup <https://github.com/NVIDIA>`_,
and `configuration guide <https://github.com/NVIDIA>`_.
Use those guides for training, debugging, evaluation and deployment instructions.
