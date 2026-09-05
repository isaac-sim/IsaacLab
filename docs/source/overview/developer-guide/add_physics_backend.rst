.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _add-physics-backend:

Add a Physics Backend
=====================

This guide extends Isaac Lab with a new physics backend. Read
:ref:`backend-architecture` for the factory-dispatch and portable-interface
design that this extension implements.

Prerequisites
-------------

Before starting, identify the engine's lifecycle, native data ownership, and
the assets and sensors it can support. The backend must provide a
:class:`~isaaclab.physics.PhysicsManager` implementation and follow the module
layout expected by :class:`~isaaclab.utils.backend_utils.FactoryBase`.

Create the backend package
--------------------------

Create an extension package using the backend name in its package path. For
example, a backend named ``mybackend`` uses this layout:

.. code-block:: text

    source/isaaclab_mybackend/
    └── isaaclab_mybackend/
        ├── __init__.py
        ├── physics/
        │   ├── __init__.py           # lazy_export()
        │   ├── __init__.pyi          # public exports
        │   ├── mybackend_manager.py
        │   └── mybackend_manager_cfg.py
        ├── assets/
        │   ├── articulation/
        │   │   ├── __init__.py
        │   │   ├── __init__.pyi
        │   │   ├── articulation.py
        │   │   └── articulation_data.py
        │   ├── rigid_object/
        │   │   └── ...
        │   ├── deformable_object/
        │   │   └── ...
        │   └── rigid_object_collection/
        │       └── ...
        ├── sensors/
        │   ├── contact_sensor/
        │   └── ...
        ├── renderers/
        │   └── ...
        └── cloner/
            └── ...

Implement the physics manager
-----------------------------

The manager must expose a :class:`~isaaclab.scene_data.SceneDataBackend` so
:class:`~isaaclab.scene_data.SceneDataProvider` can read the backend's body
transforms in the Warp-native format consumed by renderers and visualizers.

.. code-block:: python

    # isaaclab_mybackend/physics/mybackend_manager.py
    from typing import ClassVar

    from isaaclab.physics import PhysicsEvent, PhysicsManager
    from isaaclab.scene_data import SceneDataBackend, SceneDataFormat


    class MyBackendSceneDataBackend(SceneDataBackend):
        def __init__(self):
            self._scene_data = SceneDataFormat.Transform()

        @property
        def transforms(self) -> SceneDataFormat.Transform:
            # Return current world-space body transforms as a Warp transformf array.
            self._scene_data.transforms = ...  # backend-native tensor view
            return self._scene_data

        @property
        def transform_count(self) -> int:
            ...

        @property
        def transform_paths(self) -> list[str]:
            # One prim path per transform; used by SceneDataProvider.create_mapping.
            ...


    class MyBackendManager(PhysicsManager):
        _scene_data_backend: ClassVar[MyBackendSceneDataBackend | None] = None

        @classmethod
        def initialize(cls, sim_context):
            super().initialize(sim_context)
            cls._scene_data_backend = MyBackendSceneDataBackend()
            # Initialize the physics engine.

        @classmethod
        def get_scene_data_backend(cls) -> SceneDataBackend:
            return cls._scene_data_backend

        @classmethod
        def step(cls):
            # Advance simulation by one timestep.
            ...

        @classmethod
        def forward(cls):
            # Update kinematics without stepping.
            ...

        @classmethod
        def reset(cls, soft=False):
            if not soft:
                cls.dispatch_event(PhysicsEvent.PHYSICS_READY)
            # Reset simulation state.

        @classmethod
        def close(cls):
            super().close()
            # Clean up resources.

Define the physics configuration
--------------------------------

Define a :class:`~isaaclab.physics.PhysicsCfg` subclass that identifies the
manager class and holds backend-specific settings:

.. code-block:: python

    # isaaclab_mybackend/physics/mybackend_manager_cfg.py
    from isaaclab.physics import PhysicsCfg
    from isaaclab.utils.configclass import configclass


    @configclass
    class MyBackendCfg(PhysicsCfg):
        class_type = "{DIR}.mybackend_manager:MyBackendManager"
        # Backend-specific settings here.

Register the backend key
------------------------

``FactoryBase`` does not derive a new backend key from an arbitrary physics
manager class name. Add the manager prefix to the core selector in
``isaaclab.utils.backend_utils`` before factories can resolve assets and
sensors for the backend:

.. code-block:: python

    # source/isaaclab/isaaclab/utils/backend_utils.py
    @classmethod
    def _get_backend(cls, *args, **kwargs) -> str:
        from isaaclab.sim.simulation_context import SimulationContext

        manager_name = SimulationContext.instance().physics_manager.__name__.lower()
        if manager_name.startswith("mybackend"):
            return "mybackend"
        # Keep the existing newton, ovphysx, and physx cases below.

The existing ``_get_package_name()`` convention maps the ``mybackend`` key to
the ``isaaclab_mybackend`` package used above. If the integration uses another
package name, add an explicit case to ``_get_package_name()`` as well.
``FactoryBase.register()`` only caches the concrete class after this key and
package mapping has selected and imported the backend module; calling it alone
does not register a new physics-manager prefix.

Implement assets and sensors
----------------------------

Each supported asset or sensor extends the matching base class in
``isaaclab``. The implementation class name must match the factory's expected
name. Use ``lazy_export()`` in package ``__init__.py`` files. Once the core
backend-key mapping is in place, the factory imports these modules by package
and module-path convention and caches their implementation classes.

.. code-block:: python

    # isaaclab_mybackend/assets/articulation/articulation.py
    from isaaclab.assets.articulation import BaseArticulation


    class Articulation(BaseArticulation):
        def __init__(self, cfg):
            super().__init__(cfg)
            # Set up backend-specific simulation structures.

Validate backend discovery
--------------------------

``FactoryBase`` maps ``isaaclab.assets.articulation`` to
``isaaclab_mybackend.assets.articulation`` after the registered
``mybackend`` key is selected from the active physics manager. Verify the
configured discovery path with the following checklist:

- Construct the backend configuration.
- Initialize a minimal simulation with that configuration.
- Instantiate one asset supported by the backend.
- Step the simulation once.
- Close the simulation cleanly.
