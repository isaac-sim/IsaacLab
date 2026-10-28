Welcome to the bleeding edge!
=============================

Experimental features let the community try new capabilities and provide feedback
before they become part of the core packages. Expect APIs and behavior to change
as these features develop.

The ``isaaclab_contrib`` Package
---------------------------------

``source/isaaclab_contrib`` contains community-developed robots, actuators, sensors,
and controllers for specialized tasks. Contributions here are:

- Actively maintained and supported by the community
- Developed openly so the broader community can provide feedback before a wider release
- Candidates for eventual integration into the core Isaac Lab packages once they mature

The package follows Isaac Lab's standard extension structure and can be installed with optional extras that pull in
only the dependencies needed for the features you use:

.. code-block:: bash

   # The base contrib package is installed with the core Isaac Lab packages.
   # Install optional dependencies (e.g., for RLinf VLA post-training) via the
   # matching root extra. --inexact keeps the existing environment untouched.
   uv sync --inexact --extra rlinf

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
   * - **Newton Solver Coupling**
     - Proxy coupling between MJWarp and VBD. Deformable assets are implemented in
       :mod:`isaaclab_newton.assets`.
     - API reference: :mod:`~isaaclab_contrib.coupling`

Contributing
------------

We welcome contributions to ``isaaclab_contrib`` that demonstrate a framework capability missing from current
examples and have a support owner and an exit plan. Follow the
:ref:`asset and environment contribution guidelines <contributing-assets-and-environments>` before opening a pull
request. Robot ports, task variants, and application-specific workflows that reuse existing framework capabilities
are usually best maintained in a community-owned external project.
