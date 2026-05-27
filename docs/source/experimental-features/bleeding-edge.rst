Welcome to the bleeding edge!
=============================

Isaac Lab is open source because our intention is to grow a community of open collaboration for robotic simulation.
We believe that robust tools are crucial for the future of robotics.

Sometimes new features may require extensive changes to the internal structure of Isaac Lab.
Directly integrating such features before they are complete and without feedback from the full community could cause
serious issues for users caught unaware.

To address this, major and experimental features are developed in the ``isaaclab_contrib`` package and released as
experimental features. This way, the community can experiment with and contribute to a feature before it is
fully integrated, reducing the likelihood of being derailed by unexpected errors.

The ``isaaclab_contrib`` Package
---------------------------------

The ``source/isaaclab_contrib`` folder is the **community incubator** for Isaac Lab. It provides a collection of
specialized robot types, actuator models, sensors, controllers, and other components that extend the core framework
for specific use cases. Contributions here are:

- Actively maintained and supported by the community
- Developed openly so the broader community can provide feedback before a wider release
- Candidates for eventual integration into the core Isaac Lab packages once they mature

The package follows Isaac Lab's standard extension structure and can be installed with optional extras that pull in
only the dependencies needed for the features you use:

.. code-block:: bash

   # Install the base contrib package
   uv pip install -e "source/isaaclab_contrib"

   # Install with optional extras (e.g., for RLinf VLA post-training)
   uv pip install -e "source/isaaclab_contrib[rlinf]"

Current Contributions
---------------------

The following features are currently available in ``isaaclab_contrib``:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Feature
     - Description
     - Documentation
   * - **Visuo-Tactile Sensor** (TacSL)
     - GPU-accelerated simulation of vision-based tactile sensors with elastomer deformation,
       RGB tactile images, and per-taxel force fields.
     - :doc:`visuo_tactile_sensor`
   * - **RL Post-Training for VLA Models** (RLinf)
     - Reinforcement learning fine-tuning of Vision-Language-Action models (e.g., GR00T, OpenVLA)
       using RLinf's scalable PPO / Actor-Critic / SAC infrastructure.
     - :doc:`rlinf_vla_posttraining`
   * - **Multirotor Systems**
     - Full simulation support for multirotor aerial vehicles (quadcopters, hexacopters, octocopters),
       including the :class:`~isaaclab_contrib.assets.Multirotor` asset, :class:`~isaaclab_contrib.actuators.Thruster`
       actuator with asymmetric motor dynamics, and :class:`~isaaclab_contrib.mdp.actions.ThrustAction` MDP terms.
     - API reference: :mod:`~isaaclab_contrib.assets`, :mod:`~isaaclab_contrib.actuators`, :mod:`~isaaclab_contrib.mdp`
   * - **Geometric Controllers**
     - Geometric controllers for multirotor attitude, velocity, acceleration, and position tracking
       on SO(3) (Lee et al.). Suitable for both trajectory following and RL baselines.
     - API reference: :mod:`~isaaclab_contrib.controllers`
   * - **Newton VBD Deformable Objects**
     - Extended deformable object support using the Newton physics backend with Vertex Block Descent (VBD),
       including Featherstone and MjWarp coupling managers.
     - API reference: :mod:`~isaaclab_contrib.deformable`

Contributing
------------

We welcome contributions to ``isaaclab_contrib``! If you have developed specialized robot assets, novel actuator
models, custom MDP components, or domain-specific utilities, please follow the Isaac Lab contribution guidelines
and open a pull request. See the `contributing guide <https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html>`_
for more information.
