Welcome to Isaac Lab!
==========================

.. figure:: source/_static/isaaclab.jpg
   :width: 100%
   :alt: H1 Humanoid example using Isaac Lab

**Isaac Lab** is a unified and modular framework for robot learning that aims to simplify common workflows
in robotics research (such as reinforcement learning, learning from demonstrations, and motion planning). It is built on
`NVIDIA Isaac Sim`_ to leverage the latest simulation capabilities for photo-realistic scenes, and fast
and efficient simulation.

The core objectives of the framework are:

- **Modularity**: Easily customize and add new environments, robots, and sensors.
- **Agility**: Adapt to the changing needs of the community.
- **Openness**: Remain open-sourced to allow the community to contribute and extend the framework.
- **Batteries-included**: Include a number of environments, sensors, and tasks that are ready to use.

Key features available in Isaac Lab include fast and accurate physics simulation provided by PhysX,
tiled rendering APIs for vectorized rendering, domain randomization for improving robustness and adaptability,
and support for running in the cloud.

Additionally, Isaac Lab provides a variety of environments, and we are actively working on adding more environments
to the list. These include classic control tasks, fixed-arm and dexterous manipulation tasks, legged locomotion tasks,
and navigation tasks. A complete list is available in the `environments <source/overview/environments>`_ section.

Isaac lab is developed with specific robot assets that are now **Batteries-included** as part of the platform and are ready to learn! These robots include...

- **Classic** Cartpole, Humanoid, Ant
- **Fixed-Arm and Hands**: UR10, Franka, Allegro, Shadow Hand
- **Quadrupeds**: Anybotics Anymal-B, Anymal-C, Anymal-D, Unitree A1, Unitree Go1, Unitree Go2, Boston Dynamics Spot
- **Humanoids**: Unitree H1, Unitree G1
- **Quadcopter**: Crazyflie

The platform is also designed so that you can add your own robots! please refer to the
:ref:`how-to` section for details.

For more information about the framework, please refer to the `paper <https://arxiv.org/abs/2301.04195>`_
:cite:`mittal2023orbit`. For clarifications on NVIDIA Isaac ecosystem, please check out the
:doc:`/source/setup/faq` section.

.. figure:: source/_static/tasks.jpg
   :width: 100%
   :alt: Example tasks created using Isaac Lab


License
=======

The Isaac Lab framework is open-sourced under the BSD-3-Clause license.
Please refer to :ref:`license` for more details.

Acknowledgement
===============
Isaac Lab development initiated from the `Orbit <https://isaac-orbit.github.io/>`_ framework. We would appreciate if you would cite it in academic publications as well:

.. code:: bibtex

   @article{mittal2023orbit,
      author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
      journal={IEEE Robotics and Automation Letters},
      title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
      year={2023},
      volume={8},
      number={6},
      pages={3740-3747},
      doi={10.1109/LRA.2023.3270034}
   }


Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   source/setup/ecosystem
   source/setup/installation/index
   source/setup/installation/cloud_installation
   source/setup/faq

.. toctree::
   :maxdepth: 3
   :caption: Overview
   :titlesonly:

   source/overview/developer-guide/index
   source/overview/core-concepts/index
   source/overview/sensors/index
   source/overview/environments
   source/overview/reinforcement-learning/index
   source/overview/teleop_imitation
   source/overview/showroom
   source/overview/simple_agents

.. toctree::
   :maxdepth: 2
   :caption: Features

   source/features/hydra
   source/features/multi_gpu
   Tiled Rendering</source/overview/sensors/camera>
   source/features/ray
   source/features/reproducibility

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :titlesonly:

   source/tutorials/index
   source/how-to/index
   source/deployment/index

.. toctree::
   :maxdepth: 1
   :caption: Migration Guides
   :titlesonly:

   source/migration/migrating_from_isaacgymenvs
   source/migration/migrating_from_omniisaacgymenvs
   source/migration/migrating_from_orbit

.. toctree::
   :maxdepth: 1
   :caption: Source API

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: References

   source/refs/reference_architecture/index
   source/refs/additional_resources
   source/refs/contributing
   source/refs/troubleshooting
   source/refs/migration
   source/refs/issues
   source/refs/release_notes
   source/refs/changelog
   source/refs/license
   source/refs/bibliography

.. toctree::
    :hidden:
    :caption: Project Links

    GitHub <https://github.com/isaac-sim/IsaacLab>
    NVIDIA Isaac Sim <https://docs.isaacsim.omniverse.nvidia.com/latest/index.html>
    NVIDIA PhysX <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/index.html>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _NVIDIA Isaac Sim: https://docs.isaacsim.omniverse.nvidia.com/latest/index.html
