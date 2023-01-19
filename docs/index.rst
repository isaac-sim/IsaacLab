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
using the following BibTeX entry:

.. code-block:: bibtex

   @misc{mittal2023orbit,
      author = {Mayank Mittal and Calvin Yu and Qinxi Yu and Jingzhou Liu and Nikita Rudin and David Hoeller and Jia Lin Yuan and Pooria Poorsarvi Tehrani and Ritvik Singh and Yunrong Guo and Hammad Mazhar and Ajay Mandlekar and Buck Babich and Gavriel State and Marco Hutter and Animesh Garg},
      title = {ORBIT: A Unified Simulation Framework for Interactive Robot Learning Environments},
      year = {2023},
      eprint = {arXiv:2301.04195},
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
   :caption: Tutorials (beginner)

   source/tutorials/00_empty
   source/tutorials/01_arms
   source/tutorials/02_cloner
   source/tutorials/03_ik_controller
   source/tutorials/04_gym_env

.. toctree::
   :maxdepth: 2
   :caption: Source API

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: References

   source/refs/contributing
   source/refs/troubleshooting
   source/refs/changelog
   source/refs/roadmap
   source/refs/license
   source/refs/bibliography


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
