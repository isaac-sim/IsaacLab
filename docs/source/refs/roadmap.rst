.. _development_roadmap:

Development Roadmap
===================

Following is a loosely defined roadmap for the development of the codebase. The roadmap is subject to
change and is not a commitment to deliver specific features by specific dates or in the specified order.

Some of the features listed are already implemented in the codebase, but are not yet documented
and/or tested. We will be working on improving the documentation and testing of these features in the
coming months.

If you have any questions or suggestions, let us know on
`GitHub discussions <https://github.com/NVIDIA-Omniverse/Orbit/discussions>`_.

**January 2023**

* |check_|  Experimental functional API
* Supported motion generators

  * |check_| Joint-space control
  * |check_| Differential inverse kinematics control
  * |check_| Riemannian Motion Policies (RMPs)

* Supported robots

  * |check_| Quardupeds: ANYmal-B, ANYmal-C, Unitree A1
  * |check_| Arms: Franka Emika Panda, UR10
  * |check_| Mobile manipulators: Franka Emika Panda and UR10 on Clearpath Ridgeback

* Supported sensors

  * |check_| Camera (non-parallelized)
  * |check_| Height scanner (non-parallelized)

* Included environments

  * |check_| classic: MuJoCo-style environments (ant, humanoid, cartpole)
  * |check_| locomotion: flat terrain for legged robots
  * |check_| rigid-object manipulation: end-effector tracking, object lifting

**February 2023**

* |check_| Bug fixes and improvements to the functional API
* |check_| Support for `skrl <https://github.com/Toni-SM/skrl>`_ (a library for reinforcement learning)

**March 2023**

* |check_| Support for conda virtual environment
* |check_| Example of using warp-based state machine for task-space manipulation

.. attention::

    Unfortunately, due to various deadlines, the development of Orbit has been paused for the months of
    April and May. One of the many reasons for this is that we are working on a new version of Isaac Sim
    (2023.1.0) which brings in a lot of new features and improvements. We will resume the development of
    Orbit in June.

**June 2023**

* |uncheck| Example on using the APIs in an Omniverse extension
* |uncheck| Add APIs for rough terrain generation
* |uncheck| Extend MDP manager classes to use sensor observations

* Supported motion generators

  * |uncheck| Operational-space control
  * |uncheck| Model predictive control (OCS2)

* Supported sensors

  * |uncheck| Height scanner (parallelized for terrains)

* Supported robots

  * |uncheck| Quardupeds: Unitree B1, Unitree Go1
  * |uncheck| Arms: Kinova Jaco2, Kinova Gen3, Sawyer, UR10e
  * |uncheck| Mobile manipulators: Fetch, PR2

* Included environments

  * |uncheck| locomotion: rough terrain for legged robots
  * |uncheck| rigid-object manipulation: in-hand manipulation, hockey puck pushing, peg-in-hole, stacking

**July 2023**

* |uncheck| Stabilize APIs and release 1.0

* Supported sensors (depends on Isaac Sim 2023.1.0 release)

  * |uncheck| Cameras (parallelized)

**August 2023**

* Included environments

  * |uncheck| deformable-object manipulation: cloth folding, cloth lifting
  * |uncheck| deformable-object manipulation: fluid transfer, fluid pouring, soft object lifting

.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">
