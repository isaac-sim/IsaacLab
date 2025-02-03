Changelog
---------

0.10.22 (2025-01-14)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Humanoid-AMP-Dance-Direct-v0``, ``Isaac-Humanoid-AMP-Run-Direct-v0`` and ``Isaac-Humanoid-AMP-Walk-Direct-v0``
  environments as a direct RL env that implements the Humanoid AMP task.


0.10.21 (2025-01-03)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the reset of the actions in the function overriding of the low level observations of :class:`isaaclab_tasks.manager_based.navigation.mdp.PreTrainedPolicyAction`.


0.10.20 (2024-12-17)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the configuration of
  :class:`isaaclab.envs.mdp.actions.OperationalSpaceControllerAction`
  inside the ``Isaac-Reach-Franka-OSC-v0`` environment to enable nullspace control.


0.10.19 (2024-12-17)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`isaaclab_tasks.manager_based.manipulation.stack.mdp.ee_frame_pos` to output
  ``ee_frame_pos`` with respect to the environment's origin.


0.10.18 (2024-12-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Factory-Direct-v0`` environment as a direct RL env that
  implements contact-rich manipulation tasks including peg insertion,
  gear meshing, and nut threading.


0.10.17 (2024-12-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Reach-Franka-OSC-v0`` and ``Isaac-Reach-Franka-OSC-Play-v0``
  variations of the manager based reach environment that uses
  :class:`isaaclab.envs.mdp.actions.OperationalSpaceControllerAction`.


0.10.16 (2024-12-03)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` and ``Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0`` environments
  as manager-based RL envs that implement a three cube stacking task.


0.10.15 (2024-10-30)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Defined the Gymnasium task entry points with configuration strings instead of class types.
  This avoids unnecessary imports and improves the load types.
* Blacklisted ``mdp`` directories during the recursive module search.


0.10.14 (2024-10-28)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed manager-based vision cartpole environment names from Isaac-Cartpole-RGB-Camera-v0
  and Isaac-Cartpole-Depth-Camera-v0 to Isaac-Cartpole-RGB-v0 and Isaac-Cartpole-Depth-v0

0.10.13 (2024-10-28)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added feature extracted observation cartpole examples.


0.10.12 (2024-10-25)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed issues with defining Gymnasium spaces in Direct workflows due to Hydra/OmegaConf limitations with non-primitive types.


0.10.11 (2024-10-22)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Sets curriculum and commands to None in manager-based environment configurations when not needed.
  Earlier, this was done by making an empty configuration object, which is now unnecessary.


0.10.10 (2024-10-22)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the wrong selection of body id's in the :meth:`isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards.feet_slide`
  reward function. This makes sure the right IDs are selected for the bodies.


0.10.9 (2024-10-01)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` to align with Robosuite stacking env.


0.10.8 (2024-09-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Stack-Cube-Franka-IK-Rel-v0`` environment as a manager-based RL env that implements a three cube stacking task.


0.10.7 (2024-10-02)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replace deprecated :attr:`num_observations`, :attr:`num_actions` and :attr:`num_states` in single-agent direct tasks
  by :attr:`observation_space`, :attr:`action_space` and :attr:`state_space` respectively.
* Replace deprecated :attr:`num_observations`, :attr:`num_actions` and :attr:`num_states` in multi-agent direct tasks
  by :attr:`observation_spaces`, :attr:`action_spaces` and :attr:`state_space` respectively.


0.10.6 (2024-09-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Cartpole-RGB-Camera-v0`` and ``Isaac-Cartpole-Depth-Camera-v0``
  manager based camera cartpole environments.


0.10.5 (2024-09-11)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the skrl RL library integration to the latest release (skrl-v1.3.0)


0.10.4 (2024-09-10)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` environment with heterogeneous proprioception and vision observations.


0.10.3 (2024-09-05)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added environment config flag ``rerender_on_reset`` to allow updating sensor data after a reset.


0.10.2 (2024-08-23)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Shadow-Hand-Over-Direct-v0`` multi-agent environment


0.10.1 (2024-08-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Cart-Double-Pendulum-Direct-v0`` multi-agent environment

Changed
^^^^^^^

* Update skrl wrapper to support multi-agent environments.


0.10.0 (2024-08-14)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for the Hydra configuration system to all the train scripts. As a result, parameters of the environment
  and the agent can be modified using command line arguments, for example ``env.actions.joint_effort.scale=10``.


0.9.0 (2024-08-05)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced the command line input ``--cpu`` with ``--device`` in the train and play scripts. Running on cpu is
  supported by passing ``--device cpu``. Running on a specific gpu is now supported by passing ``--device cuda:<device_id>``,
  where ``<device_id>`` is the id of the GPU to use, for example ``--device cuda:0``.


0.8.2 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``Isaac-Repose-Cube-Allegro-Direct-v0`` environment

Changed
^^^^^^^

* Renamed ``Isaac-Shadow-Hand-Direct-v0`` environments to ``Isaac-Repose-Cube-Shadow-Direct-v0``.
* Renamed ``Isaac-Shadow-Hand-OpenAI-FF-Direct-v0`` environments to ``Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0``.
* Renamed ``Isaac-Shadow-Hand-OpenAI-LSTM-Direct-v0`` environments to ``Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0``.


0.8.1 (2024-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed the folder names for Unitree robots in the manager-based locomotion tasks. Earlier, there was an inconsistency
  in the folder names as some had ``unitree_`` prefix and some didn't. Now, none of the folders have the prefix.


0.8.0 (2024-07-26)
~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Renamed the action term names inside the manager-based lift-manipulation task. Earlier, they were called
  ``body_joint_pos`` and ``gripper_joint_pos``. Now, they are called ``arm_action`` and ``gripper_action``.


0.7.10 (2024-07-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Extended skrl wrapper to support training/evaluation using JAX.


0.7.9 (2024-07-01)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the action space check in the Stable-Baselines3 wrapper. Earlier, the wrapper checked
  the action space via :meth:`gymnasium.spaces.Box.is_bounded` method, which returned a bool
  value instead of a string.


0.7.8 (2024-06-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the skrl RL library integration to the latest release (>= 1.2.0)


0.7.7 (2024-06-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the tasks to use the renamed attribute :attr:`isaaclab.sim.SimulationCfg.render_interval`.


0.7.6 (2024-06-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added option to save images for Cartpole Camera environment.


0.7.5 (2024-05-31)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added exporting of empirical normalization layer to ONNX and JIT when exporting the model using
  :meth:`isaaclab.actuators.ActuatorNetMLP.export` method. Previously, the normalization layer
  was not exported to the ONNX and JIT models. This caused the exported model to not work properly
  when used for inference.


0.7.5 (2024-05-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new environment ``Isaac-Navigation-Flat-Anymal-C-v0`` to navigate towards a target position on flat terrain.


0.7.4 (2024-05-21)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Made default device for RSL RL and SB3 configs to "cuda:0".

0.7.3 (2024-05-21)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduced ``--max_iterations`` argument to training scripts for specifying number of training iterations.

0.7.2 (2024-05-13)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Shadow Hand environments: ``Isaac-Shadow-Hand-Direct-v0``, ``Isaac-Shadow-Hand-OpenAI-FF-Direct-v0``,
  and ``Isaac-Shadow-Hand-OpenAI-LSTM-Direct-v0``.


0.7.1 (2024-05-09)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the skrl agent configurations for the config and direct workflow tasks


0.7.0 (2024-05-07)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed all references of ``BaseEnv``, ``RLTaskEnv``, and ``OIGEEnv`` to
  :class:`isaaclab.envs.ManagerBasedEnv`, :class:`isaaclab.envs.ManagerBasedRLEnv`,
  and :class:`isaaclab.envs.DirectRLEnv` respectively.
* Split environments into ``manager_based`` and ``direct`` folders.

Added
^^^^^

* Added direct workflow environments:
  * ``Isaac-Cartpole-Direct-v0``, ``Isaac-Cartpole-Camera-Direct-v0``, ``Isaac-Ant-Direct-v0``, ``Isaac-Humanoid-Direct-v0``.
  * ``Isaac-Velocity-Flat-Anymal-C-Direct-v0``, ``Isaac-Velocity-Rough-Anymal-C-Direct-v0``, ``Isaac-Quadcopter-Direct-v0``.


0.6.1 (2024-04-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new environment ``Isaac-Repose-Cube-Allegro-v0`` and ``Isaac-Repose-Allegro-Cube-NoVelObs-v0``
  for the Allegro hand to reorient a cube. It is based on the IsaacGymEnvs Allegro hand environment.


0.6.0 (2024-03-10)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new environment ``Isaac-Open-Drawer-Franka-v0`` for the Franka arm to open a drawer. It is
  based on the IsaacGymEnvs cabinet environment.

Fixed
^^^^^

* Fixed logging of extra information for RL-Games wrapper. It expected the extra information to be under the
  key ``"episode"``, but Isaac Lab used the key ``"log"``. The wrapper now remaps the key to ``"episode"``.


0.5.7 (2024-02-28)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Updated the RL wrapper for the skrl library to the latest release (>= 1.1.0)


0.5.6 (2024-02-21)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the configuration parsing to support a pre-initialized configuration object.


0.5.5 (2024-02-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Pinned :mod:`torch` version to 2.0.1 in the setup.py to keep parity version of :mod:`torch` supplied by
  Isaac 2023.1.1, and prevent version incompatibility between :mod:`torch` ==2.2 and
  :mod:`typing-extensions` ==3.7.4.3


0.5.4 (2024-02-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a check for the flag :attr:`isaaclab.envs.ManagerBasedRLEnvCfg.is_finite_horizon`
  in the RSL-RL and RL-Games wrappers to handle the finite horizon tasks properly. Earlier,
  the wrappers were always assuming the tasks to be infinite horizon tasks and returning a
  time-out signals when the episode length was reached.


0.5.3 (2023-11-16)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added raising of error in the :meth:`isaaclab_tasks.utils.importer.import_all` method to make sure
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

* Fixed the wrappers to different learning frameworks to use the new :class:`isaaclab_tasks.ManagerBasedRLEnv` class.
  The :class:`ManagerBasedRLEnv` class inherits from the :class:`gymnasium.Env` class (Gym 0.29.0).
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

* Fixed the way package import automation is handled in the :mod:`isaaclab_tasks` module. Earlier it was
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

* Moved the base environment definition to the :class:`isaaclab.envs.RLEnv` class. The :class:`RLEnv`
  contains RL-specific managers such as the reward, termination, randomization and curriculum managers. These
  are all configured using the :class:`isaaclab.envs.RLEnvConfig` class. The :class:`RLEnv` class
  inherits from the :class:`isaaclab.envs.ManagerBasedEnv` and ``gym.Env`` classes.

Fixed
^^^^^

* Adapted the wrappers to use the new :class:`isaaclab.envs.RLEnv` class.


0.4.1 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Adapted the base :class:`IsaacEnv` class to use the :class:`SimulationContext` class from the
  :mod:`isaaclab.sim` module. This simplifies setting of simulation parameters.


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
* Updated the training scripts in the ``scripts/reinforcement_learning`` directory to use the new flag ``viewport``.
  If the CLI argument ``--video`` is passed, videos are recorded in the ``videos/train`` directory using the
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
* Added enabling of ``omni.kit.viewport`` and ``isaacsim.replicator`` extensions
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
