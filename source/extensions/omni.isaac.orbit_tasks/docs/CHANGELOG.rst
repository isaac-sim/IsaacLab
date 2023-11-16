Changelog
---------

0.5.3 (2023-11-16)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added raising of error in the :meth:`omni.isaac.orbit_tasks.utils.importer.import_all` method to make sure
  all the packages are imported properly. Previously, error was being caught and ignored.


0.5.2 (2023-11-08)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the RL wrappers for Stable-Baselines3 and RL-Games. It now works with their most recent versions.
* Fixed the :meth:`get_checkpoint_path` to allow any in-between sub-folders between the run directory and the
  checkpoint directory.


0.5.1 (2023-11-04)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the wrappers to different learning frameworks to use the new :class:`omni.isaac.orbit_tasks.RLTaskEnv` class.
  The :class:`RLTaskEnv` class inherits from the :class:`gymnasium.Env` class (Gym 0.29.0).
* Fixed the registration of tasks in the Gym registry based on Gym 0.29.0 API.

Changed
^^^^^^^

* Removed the inheritance of all the RL-framework specific wrappers from the :class:`gymnasium.Wrapper` class.
  This is because the wrappers don't comply with the new Gym 0.29.0 API. The wrappers are now only inherit
  from their respective RL-framework specific base classes.


0.5.0 (2023-10-30)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the way agent configs are handled for environments and learning agents. Switched from yaml to configclasses.

Fixed
^^^^^

* Fixed the way package import automation is handled in the :mod:`omni.isaac.orbit_tasks` module. Earlier it was
  not skipping the blacklisted packages properly.


0.4.3 (2023-09-25)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added future import of ``annotations`` to have a consistent behavior across Python versions.
* Removed the type-hinting from docstrings to simplify maintenance of the documentation. All type-hints are
  now in the code itself.


0.4.2 (2023-08-29)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the base environment definition to the :class:`omni.isaac.orbit.envs.RLEnv` class. The :class:`RLEnv`
  contains RL-specific managers such as the reward, termination, randomization and curriculum managers. These
  are all configured using the :class:`omni.isaac.orbit.envs.RLEnvConfig` class. The :class:`RLEnv` class
  inherits from the :class:`omni.isaac.orbit.envs.BaseEnv` and ``gym.Env`` classes.

Fixed
^^^^^

* Adapted the wrappers to use the new :class:`omni.isaac.orbit.envs.RLEnv` class.


0.4.1 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Adapted the base :class:`IsaacEnv` class to use the :class:`SimulationContext` class from the
  :mod:`omni.isaac.orbit.sim` module. This simplifies setting of simulation parameters.


0.4.0 (2023-07-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed the resetting of environment indices in the step call of the :class:`IsaacEnv` class.
  This must be handled in the :math:`_step_impl`` function by the inherited classes.
* Adapted the wrapper for RSL-RL library its new API.

Fixed
^^^^^

* Added handling of no checkpoint available error in the :meth:`get_checkpoint_path`.
* Fixed the locomotion environment for rough terrain locomotion training.


0.3.2 (2023-07-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^^^

* Added a UI to the :class:`IsaacEnv` class to enable/disable rendering of the viewport when not running in
  headless mode.

Fixed
^^^^^

* Fixed the the issue with environment returning transition tuples even when the simulation is paused.
* Fixed the shutdown of the simulation when the environment is closed.


0.3.1 (2023-06-23)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the argument ``headless`` in :class:`IsaacEnv` class to ``render``, in order to cause less confusion
  about rendering and headless-ness, i.e. that you can render while headless.


0.3.0 (2023-04-14)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new flag ``viewport`` to the :class:`IsaacEnv` class to enable/disable rendering of the viewport.
  If the flag is set to ``True``, the viewport is enabled and the environment is rendered in the background.
* Updated the training scripts in the ``source/standalone/workflows`` directory to use the new flag ``viewport``.
  If the CLI argument ``--video`` is passed, videos are recorded in the ``videos`` directory using the
  :class:`gym.wrappers.RecordVideo` wrapper.

Changed
^^^^^^^

* The :class:`IsaacEnv` class supports different rendering mode as referenced in OpenAI Gym's ``render`` method.
  These modes are:

  * ``rgb_array``: Renders the environment in the background and returns the rendered image as a numpy array.
  * ``human``: Renders the environment in the background and displays the rendered image in a window.

* Changed the constructor in the classes inheriting from :class:`IsaacEnv` to pass all the keyword arguments to the
  constructor of :class:`IsaacEnv` class.

Fixed
^^^^^

* Clarified the documentation of ``headless`` flag in the :class:`IsaacEnv` class. It refers to whether or not
  to render at every sim step, not whether to render the viewport or not.
* Fixed the unit tests for running random agent on included environments.

0.2.3 (2023-03-06)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Tuned the observations and rewards for ``Isaac-Lift-Franka-v0`` environment.

0.2.2 (2023-03-04)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the issue with rigid object not working in the ``Isaac-Lift-Franka-v0`` environment.

0.2.1 (2023-03-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a flag ``disable_contact_processing`` to the :class:`SimCfg` class to handle
  contact processing effectively when using TensorAPIs for contact reporting.
* Added verbosity flag to :meth:`export_policy_as_onnx` to print model summary.

Fixed
^^^^^

* Clarified the documentation of flags in the :class:`SimCfg` class.
* Added enabling of ``omni.kit.viewport`` and ``omni.replicator.isaac`` extensions
  dynamically to maintain order in the startup of extensions.
* Corrected the experiment names in the configuration files for training environments with ``rsl_rl``.

Changed
^^^^^^^

* Changed the default value of ``enable_scene_query_support`` in :class:`SimCfg` class to False.
  The flag is overridden to True inside :class:`IsaacEnv` class when running the simulation in
  non-headless mode.

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
