.. _migrating-from-orbit:

From Orbit
==========

.. currentmodule:: isaaclab

Since `Orbit`_ was used as basis for Isaac Lab, migrating from Orbit to Isaac Lab is straightforward.
The following sections describe the changes that need to be made to your code to migrate from Orbit to Isaac Lab.

.. note::

  The following changes are with respect to Isaac Lab 1.0 release. Please refer to the `release notes`_ for any changes
  in the future releases.


Renaming of the launch script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script ``orbit.sh`` has been renamed to ``isaaclab.sh``.


Updates to extensions
~~~~~~~~~~~~~~~~~~~~~

The extensions ``omni.isaac.orbit``, ``omni.isaac.orbit_tasks``, and ``omni.isaac.orbit_assets`` have been renamed
to ``isaaclab``, ``isaaclab_tasks``, and ``isaaclab_assets``, respectively. Thus,
the new folder structure looks like this:

- ``source/isaaclab/isaaclab``
- ``source/isaaclab_tasks/isaaclab_tasks``
- ``source/isaaclab_assets/isaaclab_assets``

The high level imports have to be updated as well:

+-------------------------------------+-----------------------------------+
| Orbit                               | Isaac Lab                         |
+=====================================+===================================+
| ``from omni.isaac.orbit...``        | ``from isaaclab...``              |
+-------------------------------------+-----------------------------------+
| ``from omni.isaac.orbit_tasks...``  | ``from isaaclab_tasks...``        |
+-------------------------------------+-----------------------------------+
| ``from omni.isaac.orbit_assets...`` | ``from isaaclab_assets...``       |
+-------------------------------------+-----------------------------------+


Updates to class names
~~~~~~~~~~~~~~~~~~~~~~

In Isaac Lab, we introduced the concept of task design workflows (see :ref:`feature-workflows`). The Orbit code is using
the manager-based workflow and the environment specific class names have been updated to reflect this change:

+------------------------+---------------------------------------------------------+
| Orbit                  | Isaac Lab                                               |
+========================+=========================================================+
| ``BaseEnv``            | :class:`isaaclab.envs.ManagerBasedEnv`                  |
+------------------------+---------------------------------------------------------+
| ``BaseEnvCfg``         | :class:`isaaclab.envs.ManagerBasedEnvCfg`               |
+------------------------+---------------------------------------------------------+
| ``RLTaskEnv``          | :class:`isaaclab.envs.ManagerBasedRLEnv`                |
+------------------------+---------------------------------------------------------+
| ``RLTaskEnvCfg``       | :class:`isaaclab.envs.ManagerBasedRLEnvCfg`             |
+------------------------+---------------------------------------------------------+
| ``RLTaskEnvWindow``    | :class:`isaaclab.envs.ui.ManagerBasedRLEnvWindow`       |
+------------------------+---------------------------------------------------------+


Updates to the tasks folder structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To support the manager-based and direct workflows, we have added two folders in the tasks extension:

- ``source/isaaclab_tasks/isaaclab_tasks/manager_based``
- ``source/isaaclab_tasks/isaaclab_tasks/direct``

The tasks from Orbit can now be found under the ``manager_based`` folder.
This change must also be reflected in the imports for your tasks. For example,

.. code-block:: python

  from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg ...

should now be:

.. code-block:: python

  from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg ...


Other Breaking changes
~~~~~~~~~~~~~~~~~~~~~~

Setting the device
------------------

The argument ``--cpu`` has been removed in favor of ``--device device_name``. Valid options for ``device_name`` are:

- ``cpu``: Use CPU.
- ``cuda``: Use GPU with device ID ``0``.
- ``cuda:N``: Use GPU, where N is the device ID. For example, ``cuda:0``.

The default value is ``cuda:0``.


Offscreen rendering
-------------------

The input argument ``--offscreen_render`` given to :class:`isaaclab.app.AppLauncher` and the environment variable
``OFFSCREEN_RENDER`` have been renamed to ``--enable_cameras`` and ``ENABLE_CAMERAS`` respectively.


Event term distribution configuration
-------------------------------------

Some of the event functions in `events.py <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/events.py>`_
accepted a ``distribution`` parameter and a ``range`` to sample from. In an effort to support arbitrary distributions,
we have renamed the input argument ``AAA_range`` to ``AAA_distribution_params`` for these functions.
Therefore, event term configurations whose functions have a ``distribution`` argument should be updated. For example,

.. code-block:: python
  :emphasize-lines: 6

  add_base_mass = EventTerm(
      func=mdp.randomize_rigid_body_mass,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="base"),
          "mass_range": (-5.0, 5.0),
          "operation": "add",
      },
  )

should now be:

.. code-block:: python
  :emphasize-lines: 6

  add_base_mass = EventTerm(
      func=mdp.randomize_rigid_body_mass,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="base"),
          "mass_distribution_params": (-5.0, 5.0),
          "operation": "add",
      },
  )


.. _Orbit: https://isaac-orbit.github.io/
.. _release notes: https://github.com/isaac-sim/IsaacLab/releases
