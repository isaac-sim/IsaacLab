Development Roadmap
===================

Following is a loosely defined roadmap for the development of the codebase. The roadmap is subject to
change and is not a commitment to deliver specific features by specific dates or in the specified order.

Some of the features listed below are already implemented in the codebase, but are not yet documented
and/or tested. We will be working on improving the documentation and testing of these features in the
coming months.

**January 2023**

* Experimental functional API
* Supported motion generators

  * Joint-space control
  * Differential inverse kinematics control
  * Riemannian Motion Policies (RMPs)

* Supported robots

  * Quardupeds: ANYmal-B, ANYmal-C, Unitree A1
  * Arms: Franka Emika Panda, UR10
  * Mobile manipulators: Franka Emika Panda and UR10 on Clearpath Ridgeback

* Supported sensors

  * Camera (non-parallelized)
  * Height scanner (non-parallelized)

* Included environments

  * classic: MuJoCo classic environments (ant, humanoid, cartpole)
  * locomotion: flat terrain for legged robots
  * rigid-object manipulation: end-effector tracking, object lifting

**February 2023**

* Example on using the APIs in an Omniverse extension
* Supported motion generators

  * Operational-space control
  * Model predictive control (OCS2)

* Supported sensors

  * Height scanner (parallelized for terrains)

* Supported robots

  * Quardupeds: Unitree B1, Unitree Go1
  * Arms: Kinova Jaco2, Kinova Gen3, Sawyer, UR10e
  * Mobile manipulators: Fetch

* Included environments

  * locomotion: rough terrain for legged robots
  * rigid-object manipulation: in-hand manipulation, hockey puck pushing, peg-in-hole, stacking
  * deformable-object manipulation: cloth folding, cloth lifting

**March or April 2023**

* Add functional versions of all environments
* Included environments

  * deformable-object manipulation: fluid transfer, fluid pouring, deformable object lifting

**May 2023**

* Update code documentation and tutorials
* Release 1.0
