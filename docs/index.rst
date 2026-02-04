Welcome to Isaac Lab!
=====================

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

The platform is also designed so that you can add your own robots! Please refer to the
:ref:`how-to` section for details.

For more information about the framework, please refer to the `technical report <https://arxiv.org/abs/2511.04831>`_
:cite:`mittal2025isaaclab`. For clarifications on NVIDIA Isaac ecosystem, please check out the
:ref:`isaac-lab-ecosystem` section.

.. figure:: source/_static/tasks.jpg
   :width: 100%
   :alt: Example tasks created using Isaac Lab


License
=======

The Isaac Lab framework is open-sourced under the BSD-3-Clause license,
with certain parts under Apache-2.0 license. Please refer to :ref:`license` for more details.

Citation
========

If you use Isaac Lab in your research, please cite our technical report:

.. code:: bibtex

   @article{mittal2025isaaclab,
      title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
      author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
      journal={arXiv preprint arXiv:2511.04831},
      year={2025},
      url={https://arxiv.org/abs/2511.04831}
   }


Acknowledgement
===============

Isaac Lab development initiated from the `Orbit <https://isaac-orbit.github.io/>`_ framework.
We gratefully acknowledge the authors of Orbit for their foundational contributions.


Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: Isaac Lab

   source/setup/ecosystem
   source/setup/installation/index
   source/deployment/index
   source/setup/installation/cloud_installation
   source/refs/reference_architecture/index


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :titlesonly:

   source/setup/quickstart
   source/overview/own-project/index
   source/setup/walkthrough/index
   source/tutorials/index
   source/how-to/index
   source/overview/developer-guide/index
   source/testing/index


.. toctree::
   :maxdepth: 3
   :caption: Overview
   :titlesonly:


   source/overview/core-concepts/index
   source/overview/environments
   source/overview/reinforcement-learning/index
   source/overview/imitation-learning/index
   source/overview/showroom
   source/overview/simple_agents


.. toctree::
   :maxdepth: 2
   :caption: Features

   source/features/hydra
   source/features/multi_gpu
   source/features/population_based_training
   Tiled Rendering</source/overview/core-concepts/sensors/camera>
   source/features/ray
   source/features/reproducibility


.. toctree::
   :maxdepth: 3
   :caption: Experimental Features

   source/experimental-features/bleeding-edge
   source/experimental-features/newton-physics-integration/index

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :titlesonly:

   source/setup/installation/cloud_installation
   source/policy_deployment/index

.. toctree::
   :maxdepth: 1
   :caption: Migration Guides
   :titlesonly:

   source/migration/migrating_to_isaaclab_3-0
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
