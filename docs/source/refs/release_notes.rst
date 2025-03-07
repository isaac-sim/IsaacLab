Release Notes
#############

The release notes are now available in the `Isaac Lab GitHub repository <https://github.com/isaac-sim/IsaacLab/releases>`_.
We summarize the release notes here for convenience.

v2.0.2
======

Overview
--------

This patch release focuses on improving actuator configuration and fixing key bugs while reverting unintended
behavioral changes from v2.0.1. **We strongly recommend switching** to this new version if you're migrating
from a pre-2.0 release of Isaac Lab.

**Key Changes:**

* **Actuator Limit Handling**: Introduced :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim`
  and :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit_sim` to clearly distinguish
  simulation solver limits from actuator model constraints. Reverted implicit actuator velocity limits
  to pre-v2.0 behavior
* **Simulation configuration update**: Removed :attr:`~isaaclab.sim.SimulationCfg.disable_contact_processing`
  flag to simplify behavior
* **Rendering configuration update**: Reverted to pre-2.0 configuration to improve the quality of the
  render product
* **Tiled camera fixes**: Fixed motion vector processing and added a hotfix for retrieving semantic
  images from the :class:`~isaaclab.sensors.TiledCamera`
* **WebRTC Support**: Added IP specification for live-streaming

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.0.1...v2.0.2

New Features
------------

* Adds :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim` and
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit_sim` to actuator.
* Adds WebRTC livestreaming support with IP specification.

Improvements
------------

* Adds guidelines and examples for code contribution
* Separates joint state setters inside Articulation class
* Implements deterministic evaluation for skrl's multi-agent algorithms
* Adds new extensions to ``pyproject.toml``
* Updates docs on Isaac Sim binary installation path and VSCode integration
* Removes remaining deprecation warning in RigidObject deprecation
* Adds security and show&tell notes to documentation
* Updates docs for segmentation and 50 series GPUs
* Adds workaround for semantic segmentation issue with tiled camera

Bug Fixes
---------

* Fixes offset from object obs for Franka stacking env when using parallel envs
* Adds scene update to ManagerBasedEnv, DirectRLEnv, and MARL envs initialization
* Loads actuator networks in eval() mode to prevent gradients
* Fixes instructions on importing ANYmal URDF in docs
* Fixes setting of root velocities in the event term :func:`~isaaclab.mdp.reset_root_state_from_terrain`
* Fixes ``activate_contact_sensors`` when using :class:`~isaaclab.sim.MultiUsdFileCfg`
* Fixes misalignment in motion vectors from :class:`~isaaclab.sim.TiledCamera`
* Sets default tensor device to CPU for Camera rot buffer

Breaking Changes
----------------

* Reverts the setting of joint velocity limits for implicit actuators
* Removes ``disable_contact_processing`` flag from SimulationContext
* Reverts to old render settings in kit experience files

Migration Guide
---------------

.. attention::

    We strongly recommend reviewing the details to fully understand the change in behavior,
    as it may impact the deployment of learned policies. Please open an issue on GitHub if
    you face any problems.


Introduction of simulation's effort and velocity limits parameters in ActuatorBaseCfg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have introduced the configuration variables :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit_sim`
and :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit_sim` to the
:class:`isaaclab.actuators.ActuatorBaseCfg` to allow users to set the **simulation** joint velocity
and effort limits through the actuator configuration class.

Previously, we were overusing the attributes :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limit`
and :attr:`~isaaclab.actuators.ActuatorBaseCfg.effort_limit` inside the actuator configuration. A series
of changes in-between led to a regression from v1.4.0 to v2.0.1 release of IsaacLab. To make this
clearer to understand, we note the change in their behavior in a tabular form:

+---------------+-------------------------+--------------------------------------------------------------------+----------------------------------------------------------------+
| Actuator Type | Attribute               | v1.4.0 Behavior                                                    | v2.0.1 Behavior                                                |
+---------------+-------------------------+--------------------------------------------------------------------+----------------------------------------------------------------+
| Implicit      | :attr:`velocity_limit`  | Ignored, not set into simulation                                   | Set into simulation                                            |
| Implicit      | :attr:`effort_limit`    | Set into simulation                                                | Set into simulation                                            |
| Explicit      | :attr:`velocity_limit`  | Used by actuator models (e.g., DC Motor), not set into simulation  | Used by actuator models (e.g., DC Motor), set into simulation  |
| Explicit      | :attr:`effort_limit`    | Used by actuator models, not set into simulation                   | Used by actuator models, set into simulation                   |
+---------------+-------------------------+--------------------------------------------------------------------+----------------------------------------------------------------+

Setting the limits from the configuration into the simulation directly affects the behavior
of the underlying physics engine solver. This impact is particularly noticeable when velocity
limits are too restrictive, especially in joints with high stiffness, where it becomes easier
to reach these limits. As a result, the change in behavior caused previously trained policies
to not function correctly in IsaacLab v2.0.1.

Consequently, we have reverted back to the prior behavior and added :attr:`velocity_limit_sim` and
:attr:`effort_limit_sim` attributes to make it clear that setting those parameters means
changing solver's configuration. The new behavior is as follows:

+----------------------------+--------------------------------------------------------+-------------------------------------------------------------+
| Attribute                  | Implicit Actuator                                      | Explicit Actuator                                           |
+============================+========================================================+=============================================================+
| :attr:`velocity_limit`     | Ignored, not set into simulation                       | Used by the model (e.g., DC Motor), not set into simulation |
| :attr:`effort_limit`       | Set into simulation (same as :attr:`effort_limit_sim`) | Used by the models, not set into simulation                 |
| :attr:`velocity_limit_sim` | Set into simulation                                    | Set into simulation                                         |
| :attr:`effort_limit_sim`   | Set into simulation (same as :attr:`effort_limit`)     | Set into simulation                                         |
+----------------------------+--------------------------------------------------------+-------------------------------------------------------------+

Users are advised to use the ``xxx_sim`` flag if they want to directly modify the solver limits.

Removal of ``disable_contact_processing`` flag in ``SimulationCfg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have now removed the ``disable_contact_processing`` flag from the :class:`isaaclab.sim.SimulationCfg`
to not have the user worry about these intricacies of the simulator. The flag is always True by
default unless a contact sensor is created (which will internally set this flag to False).

Previously, the flag ``disable_contact_processing`` led to confusion about its
behavior. As the name suggests, the flag controls the contact reporting from the
underlying physics engine, PhysX. Disabling this flag (note the double negation)
means that PhysX collects the contact information from its solver and allows
reporting them to the user. Enabling this flag means this operation is not performed and
the overhead of it is avoided.

Many of our examples (for instance, the locomotion environments) were setting this
flag to True which meant the contacts should **not** get reported. However, this issue
was not noticed earlier since GPU simulation bypasses this flag, and only CPU simulation
gets affected. Running the same examples on CPU device led to different behaviors
because of this reason.

Existing users, who currently set this flag themselves, should receive a deprecated
warning mentioning the removal of this flag and the switch to the new default behavior.

Switch to older rendering settings to improve render quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the IsaacLab 2.0.0 release, we switched to new render settings aimed at improving
tiled-rendering performance, but at the cost of reduced rendering quality. This change
particularly affected dome lighting in the scene, which is the default in many of our examples.

As reported by several users, this change negatively impacted render quality, even in
cases where it wasn’t necessary (such as when recording videos of the simulation). In
response to this feedback, we have reverted to the previous render settings by default
to restore the quality users expected.

For users looking to trade render quality for speed, we will provide guidelines in the future.


v2.0.1
======

Overview
--------

This release contains a small set of fixes and improvements.

The main change was to maintain combability with the updated library name for RSL RL, which breaks the previous installation methods for Isaac Lab. This release provides the necessary fixes and updates in Isaac Lab to accommodate for the name change and maintain combability with installation for RSL RL.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v2.0.0...v2.0.1

Improvements
------------

* Switches to RSL-RL install from PyPI by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1811
* Updates the script path in the document by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/1766
* Disables extension auto-reload when saving files by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1788
* Updates documentation for v2.0.1 installation by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1818

Bug Fixes
---------

* Fixes timestamp of com and link buffers when writing articulation pose to sim by @Jackkert in https://github.com/isaac-sim/IsaacLab/pull/1765
* Fixes incorrect local documentation preview path in xdg-open command by @louislelay in https://github.com/isaac-sim/IsaacLab/pull/1776
* Fixes no matching distribution found for rsl-rl (unavailable) by @samibouziri in https://github.com/isaac-sim/IsaacLab/pull/1808
* Fixes reset of sensor drift inside the RayCaster sensor by @zoctipus in https://github.com/isaac-sim/IsaacLab/pull/1821

New Contributors
----------------

* @Jackkert made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1765


v2.0.0
======

Overview
--------

Isaac Lab 2.0 brings some exciting new features, including a new addition to the Imitation Learning workflow with
the **Isaac Lab Mimic** extension.

Isaac Lab Mimic provides the ability to automatically generate additional trajectories based on just a few human
collected demonstrations, allowing for larger training datasets with less human effort. This work is based on the
`MimicGen <https://mimicgen.github.io/>`_ work for Scalable Robot Learning using Human Demonstrations.

Additionally, we introduced a new set of AMP tasks based on
`Adversarial Motion Priors <https://xbpeng.github.io/projects/AMP/index.html>`_, training humanoid robots to walk, run,
and dance.

Along with Isaac Lab 2.0, Isaac Sim 4.5 brings several new and breaking changes, including a full refactor of the
Isaac Sim extensions, an improved URDF importer, an update to the PyTorch dependency to version 2.5.1, and many
fixes for tiled rendering that now supports multiple tiled cameras at different resolutions.

To follow the refactoring in Isaac Sim, we made similar refactoring and restructuring changes to Isaac Lab.
These breaking changes will no longer be compatible with previous Isaac Sim versions.

.. attention::

    Please make sure to update to Isaac Sim 4.5 when using the Isaac Lab 2.0 release.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.4.1...v2.0.0

Highlights from the Isaac Sim 4.5 release
-----------------------------------------

* Support for multiple ``TiledCamera`` instances and varying resolutions
* Improved rendering performance by up to 1.2x
* Faster startup time through optimizations in the Cloner class that improves startup time by 30%
* Enhanced OmniPVD for debugging physics simulation, enabling capturing reinforcement learning simulation
* Physics simulation performance optimizations improving throughput of up to 70%
* Physics support for dedicated cylinder and cone geometry designed for robot wheels that is fully GPU accelerated
* A new physics GPU filtering mechanism allowing co-location of reinforcement learning environments at the
  origin with minimal performance loss for scenes with limited collider counts
* Improvements in simulation stability for mimic joints at high joint gains

New Features
------------

* Adds humanoid AMP tasks for direct workflow by @Toni-SM
* Adds Isaac Lab Mimic based on MimicGen data generation for Imitation Learning by @peterd-NV @nvcyc @ashwinvkNV @karsten-nvidia
* Adds consolidated demo script for showcasing recording and mimic dataset generation in real-time in one simulation script by @nvcyc
* Adds Franka stacking environment for GR00T mimic by @peterd-NV @nvcyc
* Adds option to filter collisions and real-time playback by @kellyguo11

Improvements
------------

* Adds a tutorial for policy inference in a prebuilt USD scene by @oahmednv
* Adds unit tests for multi-tiled cameras by @matthewtrepte
* Updates render setting defaults for better quality by @kellyguo11
* Adds a flag to wait for texture loading completion when reset by @oahmednv
* Adds pre-trained checkpoints and tools for generating and uploading checkpoints by @nv-cupright
* Adds new denoiser optimization flags for rendering by @kellyguo11
* Updates torch to 2.5.1 by @kellyguo11

Bug Fixes
---------

* Fixes external force buffers to set to zero when no forces/torques are applied by @matthewtrepte
* Fixes RSL-RL package name in ``setup.py`` according to PyPI installation by @samibouziri

Breaking Changes
----------------

* Updates the URDF and MJCF importers for Isaac Sim 4.5 by @Dhoeller19
* Renames Isaac Lab extensions and folders by @kellyguo11
* Restructures extension folders and removes old imitation learning scripts by @kellyguo11
* Renames default conda and venv Python environment from ``isaaclab`` to ``env_isaaclab`` by @Toni-SM

.. attention::

	We have identified a breaking feature for semantic segmentation and instance segmentation when using
	``TiledCamera`` with instanceable assets. Since the Isaac Sim 4.5 / Isaac Lab 2.0 release, semantic and instance
	segmentation outputs only render the first tile correctly and produces blank outputs for the remaining tiles.
	We will be introducing a workaround for this fix to remove scene instancing if semantic segmentation or instance
	segmentation is required for ``TiledCamera`` until we receive a proper fix from Omniverse as part of the next Isaac Sim release.

Migration Guide
---------------

Renaming of Isaac Sim Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, Isaac Sim extensions have been following the convention of ``omni.isaac.*``,
such as ``omni.isaac.core``. In Isaac Sim 4.5, Isaac Sim extensions have been renamed
to use the prefix ``isaacsim``, replacing ``omni.isaac``. In addition, many extensions
have been renamed and split into multiple extensions to prepare for a more modular
framework that can be customized by users through the use of app templates.

Notably, the following commonly used Isaac Sim extensions in Isaac Lab are renamed as follow:

* ``omni.isaac.cloner`` --> :mod:`isaacsim.core.cloner`
* ``omni.isaac.core.prims`` --> :mod:`isaacsim.core.prims`
* ``omni.isaac.core.simulation_context`` --> :mod:`isaacsim.core.api.simulation_context`
* ``omni.isaac.core.utils`` --> :mod:`isaacsim.core.utils`
* ``omni.isaac.core.world`` --> :mod:`isaacsim.core.api.world`
* ``omni.isaac.kit.SimulationApp`` --> :mod:`isaacsim.SimulationApp`
* ``omni.isaac.ui`` --> :mod:`isaacsim.gui.components`

Renaming of the URDF and MJCF Importers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from Isaac Sim 4.5, the URDF and MJCF importers have been renamed to be more consistent
with the other extensions in Isaac Sim. The importers are available on isaac-sim GitHub
as open source projects.

Due to the extension name change, the Python module names have also been changed:

* URDF Importer: :mod:`isaacsim.asset.importer.urdf` (previously :mod:`omni.importer.urdf`)
* MJCF Importer: :mod:`isaacsim.asset.importer.mjcf` (previously :mod:`omni.importer.mjcf`)

From the Isaac Sim UI, both URDF and MJCF importers can now be accessed directly from the File > Import
menu when selecting a corresponding .urdf or .xml file in the file browser.

Changes in URDF Importer
~~~~~~~~~~~~~~~~~~~~~~~~

Isaac Sim 4.5 brings some updates to the URDF Importer, with a fresh UI to allow for better configurations
when importing robots from URDF. As a result, the Isaac Lab URDF Converter has also been updated to
reflect these changes. The :class:`isaaclab.sim.converters.UrdfConverterCfg` includes some new settings,
such as :class:`~isaaclab.sim.converters.JointDriveCfg.PDGainsCfg`
and :class:`~isaaclab.sim.converters.JointDriveCfg.NaturalFrequencyGainsCfg` classes for configuring
the gains of the drives.

One breaking change to note is that the :attr:`~isaaclab.sim.converters.UrdfConverterCfg.JointDriveCfg.gains`
attribute must be of class type :class:`~isaaclab.sim.converters.JointDriveCfg.PDGainsCfg` or
:class:`~isaaclab.sim.converters.JointDriveCfg.NaturalFrequencyGainsCfg`.

The stiffness of the :class:`~isaaclab.sim.converters.JointDriveCfg.PDGainsCfg` must be specified, as such:

.. code-block:: python

    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
    )


The :attr:`~isaaclab.sim.converters.JointDriveCfg.NaturalFrequencyGainsCfg.natural_frequency` attribute must
be specified for :class:`~isaaclab.sim.converters.JointDriveCfg.NaturalFrequencyGainsCfg`.


Renaming of Isaac Lab Extensions and Folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Corresponding to Isaac Sim 4.5 changes, we have also made some updates to the Isaac Lab directories and extensions.
All extensions that were previously under ``source/extensions`` are now under the ``source/`` directory directly.
The ``source/apps`` and ``source/standalone`` folders have been moved to the root directory and are now called
``apps/`` and ``scripts/``.

Isaac Lab extensions have been renamed to:

* ``omni.isaac.lab`` --> :mod:`isaaclab`
* ``omni.isaac.lab_assets`` --> :mod:`isaaclab_assets`
* ``omni.isaac.lab_tasks`` --> :mod:`isaaclab_tasks`

In addition, we have split up the previous ``source/standalone/workflows`` directory into ``scripts/imitation_learning``
and ``scripts/reinforcement_learning`` directories. The RSL RL, Stable-Baselines, RL_Games, SKRL, and Ray directories
are under ``scripts/reinforcement_learning``, while Robomimic and the new Isaac Lab Mimic directories are under
``scripts/imitation_learning``.

To assist with the renaming of Isaac Lab extensions in your project, we have provided a
`simple script <https://gist.github.com/kellyguo11/3e8f73f739b1c013b1069ad372277a85>`_ that will traverse
through the ``source`` and ``docs`` directories in your local Isaac Lab project and replace any instance of the renamed
directories and imports. **Please use the script at your own risk as it will overwrite source files directly.**


Restructuring of Isaac Lab Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the introduction of :mod:`isaaclab_mimic`, designed for supporting data generation workflows for imitation learning,
we have also split out the previous ``wrappers`` folder under ``isaaclab_tasks`` to its own module, named :mod:`isaaclab_rl`.
This new extension will contain reinforcement learning specific wrappers for the various RL libraries supported by Isaac Lab.

The new :mod:`isaaclab_mimic` extension will also replace the previous imitation learning scripts under the ``robomimic`` folder.
We have removed the old scripts for data collection and dataset preparation in favor of the new mimic workflow. For users
who prefer to use the previous scripts, they will be available in previous release branches.

Additionally, we have also restructured the :mod:`isaaclab_assets` extension to be split into ``robots`` and ``sensors``
subdirectories. This allows for clearer separation between the pre-defined configurations provided in the extension.

As an example, the following import:

.. code-block:: python

    from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG

should be replaced with:

.. code-block:: python

    from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


v1.4.1
======

Overview
--------

This release contains a set of improvements and bug fixes.

Most importantly, we reverted one of the `changes from the previous release <https://github.com/isaac-sim/IsaacLab/pull/966>`_
to ensure the training throughput performance remains the same.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.4.0...v1.4.1

This is the **final release compatible with Isaac Sim 4.2**. The next release will target Isaac Sim 4.5,
which introduces breaking changes that will make Isaac Lab incompatible with earlier versions of Isaac Sim.

New Features
------------

* Adds documentation and demo script for IMU sensor by @mpgussert in https://github.com/isaac-sim/IsaacLab/pull/1694

Improvements
------------

* Removes deprecation for root_state_w properties and setters by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/1695
* Fixes MARL workflows for recording videos during training/inferencing by @Rishi-V in https://github.com/isaac-sim/IsaacLab/pull/1596
* Adds body tracking option to ViewerCfg by @KyleM73 in https://github.com/isaac-sim/IsaacLab/pull/1620
* Fixes the ``joint_parameter_lookup`` type in ``RemotizedPDActuatorCfg`` to support list format by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/1626
* Updates pip installation documentation to clarify options by @steple in https://github.com/isaac-sim/IsaacLab/pull/1621
* Fixes docstrings in Articulation Data that report wrong return dimension by @zoctipus in https://github.com/isaac-sim/IsaacLab/pull/1652
* Fixes documentation error for PD Actuator by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1668
* Clarifies ray documentation and fixes minor issues by @garylvov in https://github.com/isaac-sim/IsaacLab/pull/1717
* Updates code snippets in documentation to reference scripts by @mpgussert in https://github.com/isaac-sim/IsaacLab/pull/1693
* Adds dict conversion test for ActuatorBase configs by @mschweig in https://github.com/isaac-sim/IsaacLab/pull/1608

Bug Fixes
---------

* Fixes JointAction not preserving order when using all joints by @T-K-233 in https://github.com/isaac-sim/IsaacLab/pull/1587
* Fixes event term for pushing root by setting velocity by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1584
* Fixes error in Articulation where ``default_joint_stiffness`` and ``default_joint_damping`` are not correctly set for implicit actuator by @zoctipus in https://github.com/isaac-sim/IsaacLab/pull/1580
* Fixes action reset of ``pre_trained_policy_action`` in navigation environment by @nicolaloi in https://github.com/isaac-sim/IsaacLab/pull/1623
* Fixes rigid object's root com velocities timestamp check by @ori-gadot in https://github.com/isaac-sim/IsaacLab/pull/1674
* Adds interval resampling on event manager's reset call by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1750
* Corrects calculation of target height adjustment based on sensor data by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/1710
* Fixes infinite loop in ``repeated_objects_terrain`` method  by @nicolaloi in https://github.com/isaac-sim/IsaacLab/pull/1612
* Fixes issue where the indices were not created correctly for articulation setters by @AntoineRichard in https://github.com/isaac-sim/IsaacLab/pull/1660

New Contributors
~~~~~~~~~~~~~~~~

* @T-K-233 made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1587
* @steple made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1616
* @Rishi-V made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1596
* @nicolaloi made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1623
* @mschweig made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1608
* @AntoineRichard made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1660
* @ori-gadot made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1674
* @garylvov made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1717


v1.4.0
======

Overview
--------

Due to a great amount of amazing updates, we are putting out one more Isaac Lab release based off of Isaac Sim 4.2.
This release contains many great new additions and bug fixes, including several new environments, distributed training
and hyperparameter support with Ray, new live plot feature for Manager-based environments, and more.

We will now spend more focus on the next Isaac Lab release geared towards the new Isaac Sim 4.5 release coming
soon. The upcoming release will contain breaking changes in both Isaac Lab and Isaac Sim and breaks backwards
compatibility, but will come with many great fixes and improvements.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.3.0...v1.4.0

New Features
------------

* Adds Factory contact-rich manipulation tasks to IsaacLab by @noseworm in https://github.com/isaac-sim/IsaacLab/pull/1520
* Adds a Franka stacking ManagerBasedRLEnv by @peterd-NV in https://github.com/isaac-sim/IsaacLab/pull/1494
* Adds recorder manager in manager-based environments by @nvcyc in https://github.com/isaac-sim/IsaacLab/pull/1336
* Adds Ray Workflow: Multiple Run Support, Distributed Hyperparameter Tuning, and Consistent Setup Across Local/Cloud by @glvov-bdai in https://github.com/isaac-sim/IsaacLab/pull/1301
* Adds ``OperationSpaceController`` to docs and tests and implement corresponding action/action_cfg classes by @ozhanozen in https://github.com/isaac-sim/IsaacLab/pull/913
* Adds null-space control option within ``OperationSpaceController`` by @ozhanozen in https://github.com/isaac-sim/IsaacLab/pull/1557
* Adds observation term history support to Observation Manager by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/1439
* Adds live plots to managers by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/893

Improvements
------------

* Adds documentation and example scripts for sensors by @mpgussert in https://github.com/isaac-sim/IsaacLab/pull/1443
* Removes duplicated ``TerminationsCfg`` code in G1 and H1 RoughEnvCfg by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/1484
* Adds option to change the clipping behavior for all Cameras and unifies the default by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/891
* Adds check that no articulation root API is applied on rigid bodies by @lgulich in https://github.com/isaac-sim/IsaacLab/pull/1358
* Adds RayCaster rough terrain base height to reward by @Andy-xiong6 in https://github.com/isaac-sim/IsaacLab/pull/1525
* Adds position threshold check for state transitions by @DorsaRoh in https://github.com/isaac-sim/IsaacLab/pull/1544
* Adds clip range for JointAction by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/1476

Bug Fixes
---------

* Fixes noise_model initialized in direct_marl_env by @NoneJou072 in https://github.com/isaac-sim/IsaacLab/pull/1480
* Fixes entry_point and kwargs in isaaclab_tasks README by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/1485
* Fixes syntax for checking if pre-commit is installed in isaaclab.sh by @louislelay in https://github.com/isaac-sim/IsaacLab/pull/1422
* Corrects fisheye camera projection types in spawner configuration by @command-z-z in https://github.com/isaac-sim/IsaacLab/pull/1361
* Fixes actuator velocity limits propagation down the articulation root_physx_view by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/1509
* Computes Jacobian in the root frame inside the ``DifferentialInverseKinematicsAction`` class by @zoctipus in https://github.com/isaac-sim/IsaacLab/pull/967
* Adds transform for mesh_prim of ray caster sensor by @clearsky-mio in https://github.com/isaac-sim/IsaacLab/pull/1448
* Fixes configclass dict conversion for torch tensors by @lgulich in https://github.com/isaac-sim/IsaacLab/pull/1530
* Fixes error in apply_actions method in ``NonHolonomicAction`` action term. by @KyleM73 in https://github.com/isaac-sim/IsaacLab/pull/1513
* Fixes outdated sensor data after reset by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1276
* Fixes order of logging metrics and sampling commands in command manager by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1352

Breaking Changes
----------------

* Refactors pose and velocities to link frame and COM frame APIs by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/966

New Contributors
----------------

* @nvcyc made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1336
* @peterd-NV made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1494
* @NoneJou072 made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1480
* @clearsky-mio made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1448
* @Andy-xiong6 made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1525
* @noseworm made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1520

v1.3.0
======

Overview
--------

This release will be a final release based on Isaac Sim 4.2 before the transition to Isaac Sim 4.5, which will
likely contain breaking changes and no longer backwards compatible with Isaac Sim 4.2 and earlier. In this release,
we introduce many features, improvements, and bug fixes, including IMU sensors, support for various types of
gymnasium spaces, manager-based perception environments, and more.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.2.0...v1.3.0

New Features
------------

* Adds ``IMU`` sensor  by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/619
* Add Camera Benchmark Tool and Allow Correct Unprojection of distance_to_camera depth image by @glvov-bdai in https://github.com/isaac-sim/IsaacLab/pull/976
* Creates Manager Based Cartpole Vision Example Environments by @glvov-bdai in https://github.com/isaac-sim/IsaacLab/pull/995
* Adds image extracted features observation term and cartpole examples for it by @glvov-bdai in https://github.com/isaac-sim/IsaacLab/pull/1191
* Supports other gymnasium spaces in Direct workflow by @Toni-SM in https://github.com/isaac-sim/IsaacLab/pull/1117
* Adds configuration classes for spawning different assets at prim paths by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1164
* Adds a rigid body collection class by @Dhoeller19 in https://github.com/isaac-sim/IsaacLab/pull/1288
* Adds option to scale/translate/rotate meshes in the ``mesh_converter`` by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/1228
* Adds event term to randomize gains of explicit actuators by @MoreTore in https://github.com/isaac-sim/IsaacLab/pull/1005
* Adds Isaac Lab Reference Architecture documentation by @OOmotuyi in https://github.com/isaac-sim/IsaacLab/pull/1371

Improvements
------------

* Expands functionality of FrameTransformer to allow multi-body transforms by @jsmith-bdai in https://github.com/isaac-sim/IsaacLab/pull/858
* Inverts SE-2 keyboard device actions (Z, X)  for yaw command by @riccardorancan in https://github.com/isaac-sim/IsaacLab/pull/1030
* Disables backward pass compilation of warp kernels by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1222
* Replaces TensorDict with native dictionary by @Toni-SM in https://github.com/isaac-sim/IsaacLab/pull/1348
* Improves omni.isaac.lab_tasks loading time by @Toni-SM in https://github.com/isaac-sim/IsaacLab/pull/1353
* Caches PhysX view's joint paths when processing fixed articulation tendons by @Toni-SM in https://github.com/isaac-sim/IsaacLab/pull/1347
* Replaces hardcoded module paths with ``__name__`` dunder by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1357
* Expands observation term scaling to support list of floats by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/1269
* Removes extension startup messages from the Simulation App by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1217
* Adds a render config to the simulation and tiledCamera limitations to the docs by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1246
* Adds Kit command line argument support by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1293
* Modifies workflow scripts to generate random seed when seed=-1 by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1048
* Adds benchmark script to measure robot loading by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1195
* Switches from ``carb`` to ``omni.log`` for logging by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1215
* Excludes cache files from vscode explorer by @Divelix in https://github.com/isaac-sim/IsaacLab/pull/1131
* Adds versioning to the docs by @sheikh-nv in https://github.com/isaac-sim/IsaacLab/pull/1247
* Adds better error message for invalid actuator parameters by @lgulich in https://github.com/isaac-sim/IsaacLab/pull/1235
* Updates tested docker and apptainer versions for cluster deployment by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/1230
* Removes ``ml_archive`` as a dependency of ``omni.isaac.lab`` extension by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/1266
* Adds a validity check for configclasses by @Dhoeller19 in https://github.com/isaac-sim/IsaacLab/pull/1214
* Ensures mesh name is compatible with USD convention in mesh converter by @fan-ziqi in https://github.com/isaac-sim/IsaacLab/pull/1302
* Adds sanity check for the term type inside the command manager by @command-z-z in https://github.com/isaac-sim/IsaacLab/pull/1315
* Allows configclass ``to_dict`` operation to handle a list of configclasses by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/1227

Bug Fixes
---------

* Disables replicate physics for deformable teddy lift environment by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1026
* Fixes Jacobian joint indices for floating base articulations by @lorenwel in https://github.com/isaac-sim/IsaacLab/pull/1033
* Fixes setting the seed from CLI for RSL-RL by @kaixi287 in https://github.com/isaac-sim/IsaacLab/pull/1084
* Fixes camera MDP term name and reprojection docstrings by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1130
* Fixes deprecation notice for using ``pxr.Semantics`` by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1129
* Fixes scaling of default ground plane by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1133
* Fixes Isaac Sim executable on pip installation by @Toni-SM in https://github.com/isaac-sim/IsaacLab/pull/1172
* Passes device from CLI args to simulation config in standalone scripts by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/1114
* Fixes the event for randomizing rigid body material by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/1140
* Fixes the ray_caster_camera tutorial script when saving the data by @mpgussert in https://github.com/isaac-sim/IsaacLab/pull/1198
* Fixes running the docker container when the DISPLAY env variable is not defined by @GiulioRomualdi in https://github.com/isaac-sim/IsaacLab/pull/1163
* Fixes default joint pos when setting joint limits by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/1040
* Fixes device propagation for noise and adds noise tests by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/1175
* Removes additional sbatch and fixes default profile in cluster deployment by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/1229
* Fixes the checkpoint loading error in RSL-RL training script by @bearpaw in https://github.com/isaac-sim/IsaacLab/pull/1210
* Fixes pytorch broadcasting issue in ``EMAJointPositionToLimitsAction`` by @bearpaw in https://github.com/isaac-sim/IsaacLab/pull/1207
* Fixes body IDs selection when computing ``feet_slide`` reward for locomotion-velocity task by @dtc103 in https://github.com/isaac-sim/IsaacLab/pull/1277
* Fixes broken URLs in markdown files by @DorsaRoh in https://github.com/isaac-sim/IsaacLab/pull/1272
* Fixes ``net_arch`` in ``sb3_ppo_cfg.yaml`` for Isaac-Lift-Cube-Franka-v0 task by @LinghengMeng in https://github.com/isaac-sim/IsaacLab/pull/1249

New Contributors
----------------

* @riccardorancan made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1030
* @glvov-bdai made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/976
* @kaixi287 made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1084
* @lgulich made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1119
* @nv-apoddubny made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1118
* @GiulioRomualdi made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1163
* @Divelix made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1131
* @sheikh-nv made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1247
* @dtc103 made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1277
* @DorsaRoh made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1272
* @louislelay made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1271
* @LinghengMeng made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1249
* @OOmotuyi made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1337
* @command-z-z made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1315
* @MoreTore made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/1005


v1.2.0
======

Overview
--------

We leverage the new release of Isaac Sim, 4.2.0, and bring RTX-based tiled rendering, support for multi-agent
environments, and introduce many bug fixes and improvements.

Additionally, we have published an example for generating rewards using an LLM based on
`Eureka <https://github.com/eureka-research/Eureka>`_, available here: https://github.com/isaac-sim/IsaacLabEureka

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.1.0...v1.2.0

New Features
------------

* Adds RTX-based tiled rendering. This improves the overall rendering speed and quality.
* Adds the direct workflow perceptive Shadowhand Cube Repose environment ``Isaac-Repose-Cube-Shadow-Vision-Direct-v0`` by @kellyguo11.
* Adds support for multi-agent environments with the Direct workflow, with support for MAPPO and IPPO in SKRL by @Toni-SM
* Adds the direct workflow multi-agent environments ``Isaac-Cart-Double-Pendulum-Direct-v0`` and ``Isaac-Shadow-Hand-Over-Direct-v0`` by @Toni-SM
* Adds throughput benchmarking scripts for the different learning workflows by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/759
* Adds results for the benchmarks in the documentation
  `here <https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/performance_benchmarks.html>`__
  for different types of hardware by @kellyguo11
* Adds the direct workflow Allegro hand environment by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/709
* Adds video recording to the play scripts in RL workflows by @j3soon in https://github.com/isaac-sim/IsaacLab/pull/763
* Adds comparison tables for the supported RL libraries
  `here <https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_frameworks.html>`__ by @kellyguo11
* Add APIs for deformable asset by @masoudmoghani in https://github.com/isaac-sim/IsaacLab/pull/630
* Adds support for MJCF converter by @qqqwan in https://github.com/isaac-sim/IsaacLab/pull/957
* Adds a function to define camera configs through intrinsic matrix by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/617
* Adds configurable modifiers to observation manager by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/830
* Adds the Hydra configuration system for RL training by @Dhoeller19 in https://github.com/isaac-sim/IsaacLab/pull/700

Improvements
------------

* Uses PhysX accelerations for rigid body acceleration data by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/760
* Adds documentation on the frames for asset data by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/742
* Renames Unitree configs in locomotion tasks to match properly by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/714
* Adds option to set the height of the border in the ``TerrainGenerator`` by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/744
* Adds a cli arg to ``run_all_tests.py`` for testing a selected extension by @jsmith-bdai in https://github.com/isaac-sim/IsaacLab/pull/753
* Decouples rigid object and articulation asset classes by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/644
* Adds performance optimizations for domain randomization by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/494
* Allows having hybrid dimensional terms inside an observation group by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/772
* Adds a flag to preserve joint order inside ``JointActionCfg`` action term by @xav-nal in https://github.com/isaac-sim/IsaacLab/pull/787
* Adds the ability to resume training from a checkpoint with rl_games by @sizsJEon in https://github.com/isaac-sim/IsaacLab/pull/797
* Adds windows configuration to VS code tasks by @johnBuffer in https://github.com/isaac-sim/IsaacLab/pull/963
* Adapts A and D button bindings in the keyboard device by @zoctipus in https://github.com/isaac-sim/IsaacLab/pull/910
* Uses ``torch.einsum`` for  quat_rotate and quat_rotate_inverse operations by @dxyy1 in https://github.com/isaac-sim/IsaacLab/pull/900
* Expands on articulation test for multiple instances and devices by @jsmith-bdai in https://github.com/isaac-sim/IsaacLab/pull/872
* Adds setting of environment seed at initialization by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/940
* Disables default viewport when headless but cameras are enabled by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/851
* Simplifies the return type for ``parse_env_cfg`` method by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/965
* Simplifies the if-elses inside the event manager apply method by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/948

Bug Fixes
---------

* Fixes rendering frame delays. Rendered images now faithfully represent the latest state of the physics scene. We added the flag
  ``rerender_on_reset`` in the environment configs to toggle an additional render step when a reset happens. When activated, the images/observation always represent the latest state of the environment, but this also reduces performance.
* Fixes ``wrap_to_pi`` function in math utilities by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/771
* Fixes setting of pose when spawning a mesh by @masoudmoghani in https://github.com/isaac-sim/IsaacLab/pull/692
* Fixes caching of the terrain using the terrain generator by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/757
* Fixes running train scripts when rsl_rl is not installed by @Dhoeller19 in https://github.com/isaac-sim/IsaacLab/pull/784, https://github.com/isaac-sim/IsaacLab/pull/789
* Adds flag to recompute inertia when randomizing the mass of a rigid body by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/989
* Fixes support for ``classmethod`` when defining a configclass by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/901
* Fixes ``Sb3VecEnvWrapper`` to clear buffer on reset by @EricJin2002 in https://github.com/isaac-sim/IsaacLab/pull/974
* Fixes venv and conda pip installation on windows by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/970
* Sets native livestream extensions to Isaac Sim 4.1-4.0 defaults by @jtigue-bdai in https://github.com/isaac-sim/IsaacLab/pull/954
* Defaults the gym video recorder fps to match episode decimation by @ozhanozen in https://github.com/isaac-sim/IsaacLab/pull/894
* Fixes the event manager's apply method by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/936
* Updates camera docs with world units and introduces new test for intrinsics by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/886
* Adds the ability to resume training from a checkpoint with rl_games by @sizsJEon in https://github.com/isaac-sim/IsaacLab/pull/797

Breaking Changes
----------------

* Simplifies device setting in SimulationCfg and AppLauncher by @Dhoeller19 in https://github.com/isaac-sim/IsaacLab/pull/696
* Fixes conflict in teleop-device command line argument in scripts by @Dhoeller19 in https://github.com/isaac-sim/IsaacLab/pull/791
* Converts container.sh into Python utilities by @hhansen-bdai  in https://github.com/isaac-sim/IsaacLab/commit/f565c33d7716db1be813b30ddbcf9321712fc497
* Drops support for ``TiledCamera`` for Isaac Sim 4.1

Migration Guide
---------------

Setting the simulation device into the simulation context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, changing the simulation device to CPU required users to set other simulation parameters (such as disabling GPU physics and GPU pipelines). This made setting up the device appear complex. We now simplify the checks for device directly inside the simulation context, so users only need to specify the device through the configuration object.

Before:

.. code:: python

    sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False, dt=0.01, physx=sim_utils.PhysxCfg(use_gpu=False))

Now:

.. code:: python

    sim_utils.SimulationCfg(device="cpu", dt=0.01, physx=sim_utils.PhysxCfg())

Setting the simulation device from CLI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, users could specify the device through the command line argument ``--device_id``. However, this made it ambiguous when users wanted to set the device to CPU. Thus, instead of the device ID, users need to specify the device explicitly through the argument ``--device``. The valid options for the device name are:

* "cpu": runs simulation on CPU
* "cuda": runs simulation on GPU with device ID at default index
* "cuda:N": runs simulation on GPU with device ID at ``N``. For instance, "cuda:0" will use device at index "0".

Due to the above change, the command line interaction with some of the scripts has changed.

Before:

.. code:: bash

    ./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task Isaac-Cartpole-v0 --headless --cpu

Now:

.. code:: bash

    ./isaaclab.sh -p source/standalone/workflows/sb3/train.py --task Isaac-Cartpole-v0 --headless --device cpu

Renaming of teleoperation device CLI in standalone scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since ``--device`` is now an argument provided by the AppLauncher, it conflicted with the command-line argument used for specifying the teleoperation-device in some of the standalone scripts. Thus, to fix this conflict, the teleoperation-device now needs to be specified through ``--teleop_device`` argument.

Before:

.. code:: bash

    ./isaaclab.sh -p source/standalone/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --device keyboard

Now:

.. code:: bash

    ./isaaclab.sh -p source/standalone/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --teleop_device keyboard


Using Python-version of container utility script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prior `container.sh <https://github.com/isaac-sim/IsaacLab/blob/v1.1.0/docker/container.sh>`_ became quite complex as it had many different use cases in one script. For instance, building a docker image for "base" or "ros2", as well as cluster deployment. As more users wanted to have the flexibility to overlay their own docker settings, maintaining this bash script became cumbersome. Hence, we migrated its features into a Python script in this release. Additionally, we split the cluster-related utilities into their own script inside the ``docker/cluster`` directory.

We still maintain backward compatibility for ``container.sh``. Internally, it calls the Python script ``container.py``. We request users to use the Python script directly.

Before:

.. code:: bash

    ./docker/container.sh start


Now:

.. code:: bash

    ./docker/container.py start


Using separate directories for logging videos in RL workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, users could record videos during the RL training by specifying the ``--video`` flag to the ``train.py`` script. The videos would be saved inside the ``videos`` directory in the corresponding log directory of the run.

Since many users requested to also be able to record videos while inferencing the policy, recording videos have also been added to the ``play.py`` script. Since changing the prefix of the video file names is not possible, the videos from the train and play scripts are saved inside the ``videos/train`` and ``videos/play`` directories, respectively.

Drops support for the tiled camera with Isaac Sim 4.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Various fixes have been made to the tiled camera rendering pipeline in Isaac Sim 4.2. This made supporting the tiled camera with Isaac Sim 4.1 difficult. Hence, for the best experience, we advice switching to Isaac Sim 4.2 with this release of Isaac Lab.

New Contributors
----------------

* @xav-nal made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/787
* @sizsJEon made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/797
* @jtigue-bdai made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/830
* @StrainFlow made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/835
* @mpgussert made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/827
* @Symars made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/898
* @martinmatak made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/876
* @bearpaw made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/945
* @dxyy1 made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/900
* @qqqwan made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/957
* @johnBuffer made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/963
* @EricJin2002 made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/974
* @iamnambiar made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/986

v1.1.0
======

Overview
--------

With the release of Isaac Sim 4.0 and 4.1, support for Isaac Sim 2023.1.1 has been discontinued.
We strongly encourage all users to upgrade to Isaac Sim 4.1 to take advantage of the latest features
and improvements. For detailed information on this upgrade, please refer to the release notes available
`here <https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html#>`__.

Besides the above, the Isaac Lab release brings new features and improvements, as detailed below. We thank
all our contributors for their continued support.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v1.0.0...v1.1.0

New Features
------------

* Adds distributed multi-GPU learning support for skrl by @Toni-SM in https://github.com/isaac-sim/IsaacLab/pull/574
* Updates skrl integration to support training/evaluation using JAX by @Toni-SM in https://github.com/isaac-sim/IsaacLab/pull/592
* Adds lidar pattern for raycaster sensor by @pascal-roth in https://github.com/isaac-sim/IsaacLab/pull/616
* Adds support for PBS job scheduler-based clusters by @shafeef901 in https://github.com/isaac-sim/IsaacLab/pull/605
* Adds APIs for spawning deformable meshes by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/613

Improvements
------------

* Changes documentation color to the green theme by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/585
* Fixes sphinx tabs to make them work in dark theme by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/584
* Fixes VSCode settings to work with pip installation of Isaac Sim by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/628
* Fixes ``isaaclab`` scripts to deal with Isaac Sim pip installation by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/631
* Optimizes interactive scene for homogeneous cloning by @kellyguo11 in https://github.com/isaac-sim/IsaacLab/pull/636
* Improves docker X11 forwarding documentation by @j3soon in https://github.com/isaac-sim/IsaacLab/pull/685

Bug Fixes
---------

* Reads gravity direction from simulation inside ``RigidObjectData`` by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/582
* Fixes reference count in asset instances due to circular references by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/580
* Fixes issue with asset deinitialization due to torch > 2.1 by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/640
* Fixes the rendering logic regression in environments by @Dhoeller19 in https://github.com/isaac-sim/IsaacLab/pull/614
* Fixes the check for action-space inside Stable-Baselines3 wrapper by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/610
* Fixes warning message in Articulation config processing by @locoxsoco in https://github.com/isaac-sim/IsaacLab/pull/699
* Fixes action term in the reach environment by @masoudmoghani in https://github.com/isaac-sim/IsaacLab/pull/710
* Fixes training UR10 reach with RL_GAMES and SKRL by @sudhirpratapyadav in https://github.com/isaac-sim/IsaacLab/pull/678
* Adds event manager call to simple manage-based env by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/666

Breaking Changes
----------------

* Drops official support for Isaac Sim 2023.1.1
* Removes the use of body view inside the asset classes by @Mayankm96 in https://github.com/isaac-sim/IsaacLab/pull/643
* Renames ``SimulationCfg.substeps`` to ``SimulationCfg.render_interval`` by @Dhoeller19 in https://github.com/isaac-sim/IsaacLab/pull/515

Migration Guide
---------------

Renaming of ``SimulationCfg.substeps``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, the users set both ``omni.isaac.lab.sim.SimulationCfg.dt`` and ``omni.isaac.lab.sim.SimulationCfg.substeps``, which marked the physics insulation time-step and sub-steps, respectively. It was unclear whether sub-steps meant the number of integration steps inside the physics time-step ``dt`` or the number of physics steps inside a rendering step.

Since in the code base, the attribute was used as the latter, it has been renamed to ``render_interval`` for clarity.

Removal of Deprecated Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As notified in previous releases, we removed the classes and attributes marked as deprecated. These are as follows:

* The ``mdp.add_body_mass`` method in the events. Please use the ``mdp.randomize_rigid_body_mass`` instead.
* The classes ``managers.RandomizationManager`` and ``managers.RandomizationTermCfg``. Please use the
  ``managers.EventManager`` and ``managers.EventTermCfg`` classes instead.
* The following properties in ``omni.isaac.lab.sensors.FrameTransformerData``:
  * ``target_rot_source`` --> ``target_quat_w``
  * ``target_rot_w`` --> ``target_quat_source``
  * ``source_rot_w`` --> ``source_quat_w``

* The attribute ``body_physx_view`` from the ``omni.isaac.lab.assets.Articulation`` and
  ``omni.isaac.lab.assets.RigidObject`` classes. These caused confusion when used with the articulation view
  since the body names did not follow the same ordering.

v1.0.0
======

Overview
--------

Welcome to the first official release of Isaac Lab!

Building upon the foundation of the `Orbit <https://isaac-orbit.github.io/>`_ framework, we have integrated
the RL environment designing workflow from `OmniIsaacGymEnvs <https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs>`_.
This allows users to choose a suitable `task-design approach <https://isaac-sim.github.io/IsaacLab/source/features/task_workflows.html>`_
for their applications.

While we maintain backward compatibility with Isaac Sim 2023.1.1, we highly recommend using Isaac Lab with
Isaac Sim 4.0.0 version for the latest features and improvements.

**Full Changelog**: https://github.com/isaac-sim/IsaacLab/compare/v0.3.1...v1.0.0

New Features
------------

* Integrated CI/CD pipeline, which is triggered on pull requests and publishes the results publicly
* Extended support for Windows OS platforms
* Added `tiled rendered <https://isaac-sim.github.io/IsaacLab/source/features/tiled_rendering.html>`_ based Camera sensor implementation. This provides optimized RGB-D rendering throughputs of up to 10k frames per second.
* Added support for multi-GPU and multi-node training for the RL-Games library
* Integrated APIs for environment designing (direct workflow) without relying on managers
* Added implementation of delayed PD actuator model
* `Added various new learning environments <https://isaac-sim.github.io/IsaacLab/main/source/features/environments.html>`_:
  * Cartpole balancing using images
  * Shadow hand cube reorientation
  * Boston Dynamics Spot locomotion
  * Unitree H1 and G1 locomotion
  * ANYmal-C navigation
  * Quadcopter target reaching

Improvements
------------

* Reduced start-up time for scripts (inherited from Isaac Sim 4.0 improvements)
* Added lazy buffer implementation for rigid object and articulation data. Instead of updating all the quantities at every step call, the lazy buffers are updated only when the user queries them
* Added SKRL support to more environments

Breaking Changes
----------------

For users coming from Orbit, this release brings certain breaking changes. Please check the migration guide for more information.

Migration Guide
---------------

Please find detailed migration guides as follows:

* `From Orbit to IsaacLab <https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_orbit.html>`_
* `From OmniIsaacGymEnvs to IsaacLab <https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_omniisaacgymenvs.html>`_

New Contributors
----------------

* @abizovnuralem made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/452
* @eltociear made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/460
* @zoctipus made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/486
* @JunghwanRo made their first contribution in https://github.com/isaac-sim/IsaacLab/pull/497

.. _simple script: https://gist.github.com/kellyguo11/3e8f73f739b1c013b1069ad372277a85
