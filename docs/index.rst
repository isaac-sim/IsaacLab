Overview
========

**Isaac Orbit** (or *orbit* in short) is a unified and modular framework, built on top of `NVIDIA
Omniverse <https://docs.omniverse.nvidia.com/>`__ and `Isaac
Sim <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html>`__,
for robot learning. It offers a modular design to easily and efficiently
create robot learning environments with photo-realistic scenes, and fast
and efficient simulation.

.. figure:: source/_static/tasks.jpg
   :width: 100%
   :alt: Example tasks created using orbit


If you use ``orbit`` in your work, please cite the `paper <https://arxiv.org/abs/2301.04195>`_
:cite:`mittal2023orbit` using the following BibTeX entry:

.. code-block:: bibtex

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

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   source/setup/installation
   source/setup/developer
   source/setup/sample

.. toctree::
   :maxdepth: 2
   :caption: Features

   source/features/environments
   source/features/actuators
   source/features/motion_generators

.. toctree::
   :maxdepth: 1
   :caption: Tutorials (Core)

   source/tutorials/00_empty
   source/tutorials/01_arms
   source/tutorials/02_cloner
   source/tutorials/03_ik_controller

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials (Environments)

   source/tutorials_envs/00_gym_env
   source/tutorials_envs/01_create_env
   source/tutorials_envs/02_wrappers

.. toctree::
   :maxdepth: 2
   :caption: Source API

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: References

   source/refs/faq
   source/refs/contributing
   source/refs/troubleshooting
   source/refs/issues
   source/refs/changelog
   source/refs/roadmap
   source/refs/license
   source/refs/bibliography


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
