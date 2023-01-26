Changelog
---------

0.2.0 (2023-01-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added environment wrapper and sequential trainer for the skrl RL library
* Added training/evaluation configuration files for the skrl RL library

0.1.2 (2023-01-19)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added the flag ``replicate_physics`` to the :class:`SimCfg` class.
* Increased the default value of ``gpu_found_lost_pairs_capacity`` in :class:`PhysxCfg` class

0.1.1 (2023-01-18)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a bug in ``Isaac-Velocity-Anymal-C-v0`` where the domain randomization is
  not applicable if cloning the environments with ``replicate_physics=True``.

0.1.0 (2023-01-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the extension.
* Includes the following environments:

  * ``Isaac-Cartpole-v0``: A cartpole environment with a continuous action space.
  * ``Isaac-Ant-v0``: A 3D ant environment with a continuous action space.
  * ``Isaac-Humanoid-v0``: A 3D humanoid environment with a continuous action space.
  * ``Isaac-Reach-Franka-v0``: A end-effector pose tracking task for the Franka arm.
  * ``Isaac-Lift-Franka-v0``: A 3D object lift and reposing task for the Franka arm.
  * ``Isaac-Velocity-Anymal-C-v0``: An SE(2) velocity tracking task for legged robot on flat terrain.
