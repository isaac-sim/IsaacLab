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


Renaming of omni.isaac.core Classes
-----------------------------------

Isaac Sim 4.5 introduced some naming changes to the core prim classes that are commonly
used in Isaac Lab. These affect the single and ``View`` variations of the prim classes, including
Articulation, RigidPrim, XFormPrim, and others. Single-object classes are now prefixed with
``Single``, such as ``SingleArticulation``, while tensorized View classes now have the ``View``
suffix removed.

The exact renamings of the classes are as follow:

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
