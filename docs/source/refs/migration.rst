.. _migration_guide:

Migration Guide (Isaac Sim)
===========================

Moving from Isaac Sim 4.2 to 4.5 and later brings in a number of changes to the
APIs and Isaac Sim extensions and classes. This document outlines the changes
and how to migrate your code to the new APIs.


Renaming of Isaac Sim Extensions
--------------------------------

Previously, Isaac Sim extensions have been following the convention of ``omni.isaac.*``,
such as ``omni.isaac.core``. In Isaac Sim 4.5, Isaac Sim extensions have been renamed
to use the prefix ``isaacsim``, replacing ``omni.isaac``. In addition, many extensions
have been renamed and split into multiple extensions to prepare for a more modular
framework that can be customized by users through the use of app templates.

Notably, the following commonly used Isaac Sim extensions in Isaac Lab are renamed as follow:

* ``omni.isaac.cloner`` --> ``isaacsim.core.cloner``
* ``omni.isaac.core.prims`` --> ``isaacsim.core.prims``
* ``omni.isaac.core.simulation_context`` --> ``isaacsim.core.api.simulation_context``
* ``omni.isaac.core.utils`` --> ``isaacsim.core.utils``
* ``omni.isaac.core.world`` --> ``isaacsim.core.api.world``
* ``omni.isaac.kit.SimulationApp`` --> ``isaacsim.SimulationApp``
* ``omni.isaac.ui`` --> ``isaacsim.gui.components``


Renaming of the URDF and MJCF Importers
---------------------------------------

Starting from Isaac Sim 4.5, the URDF and MJCF importers have been renamed to be more consistent
with the other extensions in Isaac Sim. The importers are available on isaac-sim GitHub
as open source projects.

Due to the extension name change, the Python module names have also been changed:

* URDF Importer: :mod:`isaacsim.asset.importer.urdf` (previously :mod:`omni.importer.urdf`)
* MJCF Importer: :mod:`isaacsim.asset.importer.mjcf` (previously :mod:`omni.importer.mjcf`)

From the Isaac Sim UI, both URDF and MJCF importers can now be accessed directly from the File > Import
menu when selecting a corresponding .urdf or .xml file in the file browser.


Changes in URDF Importer
------------------------

Isaac Sim 4.5 brings some updates to the URDF Importer, with a fresh UI to allow for better configurations
when importing robots from URDF. As a result, the Isaac Lab URDF Converter has also been updated to
reflect these changes. The :class:`UrdfConverterCfg` includes some new settings, such as :class:`PDGainsCfg`
and :class:`NaturalFrequencyGainsCfg` classes for configuring the gains of the drives.

One breaking change to note is that the :attr:`UrdfConverterCfg.JointDriveCfg.gains` attribute must
be of class type :class:`PDGainsCfg` or :class:`NaturalFrequencyGainsCfg`.

The stiffness of the :class:`PDGainsCfg` must be specified, as such:

.. code::python

    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
    )

The :attr:`natural_frequency` must be specified for :class:`NaturalFrequencyGainsCfg`.


Renaming of omni.isaac.core Classes
-----------------------------------

Isaac Sim 4.5 introduced some naming changes to the core prim classes that are commonly
used in Isaac Lab. These affect the single and ``View`` variations of the prim classes, including
Articulation, RigidPrim, XFormPrim, and others. Single-object classes are now prefixed with
``Single``, such as ``SingleArticulation``, while tensorized View classes now have the ``View``
suffix removed.

The exact renaming of the classes are as follow:

* ``Articulation`` --> ``SingleArticulation``
* ``ArticulationView`` --> ``Articulation``
* ``ClothPrim`` --> ``SingleClothPrim``
* ``ClothPrimView`` --> ``ClothPrim``
* ``DeformablePrim`` --> ``SingleDeformablePrim``
* ``DeformablePrimView`` --> ``DeformablePrim``
* ``GeometryPrim`` --> ``SingleGeometryPrim``
* ``GeometryPrimView`` --> ``GeometryPrim``
* ``ParticleSystem`` --> ``SingleParticleSystem``
* ``ParticleSystemView`` --> ``ParticleSystem``
* ``RigidPrim`` --> ``SingleRigidPrim``
* ``RigidPrimView`` --> ``RigidPrim``
* ``XFormPrim`` --> ``SingleXFormPrim``
* ``XFormPrimView`` --> ``XFormPrim``


Renaming of Isaac Lab Extensions and Folders
--------------------------------------------

Corresponding to Isaac Sim 4.5 changes, we have also made some updates to the Isaac Lab directories and extensions.
All extensions that were previously under ``source/extensions`` are now under the ``source/`` directory directly.
The ``source/apps`` and ``source/standalone`` folders have been moved to the root directory and are now called
``apps/`` and ``scripts/``.

Isaac Lab extensions have been renamed to:

* ``omni.isaac.lab`` --> ``isaaclab``
* ``omni.isaac.lab_assets`` --> ``isaaclab_assets``
* ``omni.isaac.lab_tasks`` --> ``isaaclab_tasks``

In addition, we have split up the previous ``source/standalone/workflows`` directory into ``scripts/imitation_learning``
and ``scripts/reinforcement_learning`` directories. The RSL RL, Stable-Baselines, RL_Games, SKRL, and Ray directories
are under ``scripts/reinforcement_learning``, while Robomimic and the new Isaac Lab Mimic directories are under
``scripts/imitation_learning``.

To assist with the renaming of Isaac Lab extensions in your project, we have provided a `simple script`_ that will traverse
through the ``source`` and ``docs`` directories in your local Isaac Lab project and replace any instance of the renamed
directories and imports. **Please use the script at your own risk as it will overwrite source files directly.**


Restructuring of Isaac Lab Extensions
-------------------------------------

With the introduction of ``isaaclab_mimic``, designed for supporting data generation workflows for imitation learning,
we have also split out the previous ``wrappers`` folder under ``isaaclab_tasks`` to its own module, named ``isaaclab_rl``.
This new extension will contain reinforcement learning specific wrappers for the various RL libraries supported by Isaac Lab.

The new ``isaaclab_mimic`` extension will also replace the previous imitation learning scripts under the ``robomimic`` folder.
We have removed the old scripts for data collection and dataset preparation in favor of the new mimic workflow. For users
who prefer to use the previous scripts, they will be available in previous release branches.

Additionally, we have also restructured the ``isaaclab_assets`` extension to be split into ``robots`` and ``sensors``
subdirectories. This allows for clearer separation between the pre-defined configurations provided in the extension.
For any existing imports such as ``from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG``, please replace it with
``from isaaclab.robots.anymal import ANYMAL_C_CFG``.


Lazy Exporting and Resolvable Strings
--------------------------------------

Isaac Lab now uses **lazy exporting** throughout all packages so that importing a top-level
module (e.g. ``import isaaclab.sensors``) no longer eagerly pulls in heavyweight
dependencies such as ``pxr``, ``omni``, or ``scipy``. This is critical because Kit and the
Isaac Sim viewer do **not** tolerate imports of ``pxr``, ``omni``, or ``scipy`` before the
application is launched — doing so will cause crashes or undefined behavior. With lazy
exporting, config objects can be constructed *before* ``SimulationApp`` is launched, which
enables automatic physics-backend selection without requiring flags like
``--enable_cameras``.

Two key patterns support this:

1. **Lazy exports** — Every ``__init__.py`` uses :func:`~isaaclab.utils.module.lazy_export`
   together with an adjacent ``.pyi`` stub to defer submodule and symbol imports until
   first access.
2. **Resolvable strings** — Config fields such as ``class_type`` store implementation
   references as strings (e.g. ``"{DIR}.sensor:Sensor"``) instead of direct class imports.
   The string is resolved to the actual class only after ``SimulationApp`` has been
   initialized.

For full details, examples, and the ``{DIR}`` placeholder convention, see the
:ref:`contributing` guide — in particular the
`Lazy Loading & Module Exports <contributing.html#lazy-loading-module-exports>`__,
`Resolvable Strings <contributing.html#resolvable-strings>`__, and
`Config + Implementation File Split <contributing.html#config-implementation-file-split>`__
sections.

Lazy Exporting in User Code
----------------------------

If your own project imports Isaac Lab symbols eagerly (i.e. via normal ``from ... import``
statements in ``__init__.py``), those imports may trigger heavyweight modules before the
simulation app is ready. This prevents automatic backend selection and may require you to
pass explicit flags like ``--enable_cameras`` or ``--kit``.

To fix this, adopt the same lazy-exporting pattern used throughout Isaac Lab:

1. Rename your existing ``__init__.py`` to ``__init__.pyi`` (this becomes the type stub).
2. Create a new ``__init__.py`` that calls ``lazy_export()``:

.. code:: python

   # my_package/__init__.py
   from isaaclab.utils.module import lazy_export

   lazy_export()

3. Ensure the ``.pyi`` stub uses **relative imports** and declares ``__all__``:

.. code:: python

   # my_package/__init__.pyi
   __all__ = ["MyCfg", "MyClass"]

   from .my_cfg import MyCfg
   from .my_class import MyClass

With this in place, ``import my_package`` will not eagerly import any submodules. Symbols
are loaded on first access, giving ``SimulationApp`` time to initialize and auto-detect the
correct backend.

For more details, refer to the :ref:`contributing` guide.


.. _simple script: https://gist.github.com/kellyguo11/3e8f73f739b1c013b1069ad372277a85
