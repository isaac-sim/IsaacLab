Isaac Lab API Documentation
===========================

.. figure:: source/_static/isaaclab.jpg
   :width: 100%
   :alt: H1 Humanoid example using Isaac Lab

**Isaac Lab** is a unified and modular framework for robot learning that aims to simplify common workflows
in robotics research (such as reinforcement learning, learning from demonstrations, and motion planning). It is built on
`NVIDIA Isaac Sim`_ to leverage the latest simulation capabilities for photo-realistic scenes, and fast
and efficient simulation.

License
=======

The Isaac Lab framework is open-sourced under the BSD-3-Clause license.

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


API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Source API

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: References

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
