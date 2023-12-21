Overview
========

**Orbit** is a unified and modular framework for robot learning that aims to simplify common workflows
in robotics research (such as RL, learning from demonstrations, and motion planning). It is built upon
`NVIDIA Isaac Sim`_ to leverage the latest simulation capabilities for photo-realistic scenes, and fast
and efficient simulation. The core objectives of the framework are:

- **Modularity**: Easily customize and add new environments, robots, and sensors.
- **Agility**: Adapt to the changing needs of the community.
- **Openness**: Remain open-sourced to allow the community to contribute and extend the framework.
- **Battery-included**: Include a number of environments, sensors, and tasks that are ready to use.

For more information about the framework, please refer to the `paper <https://arxiv.org/abs/2301.04195>`_
:cite:`mittal2023orbit`. For clarifications on NVIDIA Isaac ecosystem, please check out the
:doc:`/source/refs/faq` section.

.. figure:: source/_static/tasks.jpg
   :width: 100%
   :alt: Example tasks created using orbit


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
   .. source/features/motion_generators

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :titlesonly:

   source/tutorials/index
   source/how-to/index
   source/deployment/index

.. toctree::
   :maxdepth: 1
   :caption: Source API

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: References

   source/refs/faq
   source/refs/migration
   source/refs/contributing
   source/refs/troubleshooting
   source/refs/issues
   source/refs/changelog
   source/refs/license
   source/refs/bibliography


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _NVIDIA Isaac Sim: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html
