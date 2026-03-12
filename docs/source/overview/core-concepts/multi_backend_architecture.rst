Multi-Backend Architecture
==========================

Isaac Lab 3.0 introduced a multi-backend architecture that enables running simulations with
different physics engines (PhysX and Newton) while maintaining a unified API. This page explains
how the backend system works and how to extend it.

Overview
--------

Instead of hard-coding a single physics engine, Isaac Lab uses a **factory pattern** to
dispatch object creation to backend-specific implementations at runtime. When you write:

.. code-block:: python

    from isaaclab.assets import Articulation

    robot = Articulation(cfg)

The ``Articulation`` class is a factory that automatically creates a
:class:`~isaaclab_physx.assets.articulation.Articulation` or
:class:`~isaaclab_newton.assets.articulation.Articulation` depending on which physics backend
is active. Your code never needs to import backend-specific modules directly.

This pattern applies to all simulation components:

.. list-table::
   :header-rows: 1

   * - Component
     - Core API (``isaaclab``)
     - PhysX (``isaaclab_physx``)
     - Newton (``isaaclab_newton``)
   * - Physics Manager
     - :class:`~isaaclab.physics.PhysicsManager`
     - :class:`~isaaclab_physx.physics.PhysxManager`
     - :class:`~isaaclab_newton.physics.NewtonManager`
   * - Articulation
     - :class:`~isaaclab.assets.Articulation`
     - :class:`~isaaclab_physx.assets.Articulation`
     - :class:`~isaaclab_newton.assets.Articulation`
   * - Rigid Object
     - :class:`~isaaclab.assets.RigidObject`
     - :class:`~isaaclab_physx.assets.RigidObject`
     - :class:`~isaaclab_newton.assets.RigidObject`
   * - Contact Sensor
     - :class:`~isaaclab.sensors.ContactSensor`
     - :class:`~isaaclab_physx.sensors.ContactSensor`
     - :class:`~isaaclab_newton.sensors.ContactSensor`
   * - Renderer
     - :class:`~isaaclab.renderers.Renderer`
     - :class:`~isaaclab_physx.renderers.IsaacRtxRenderer`
     - :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`
   * - Scene Data Provider
     - :class:`~isaaclab.physics.SceneDataProvider`
     - :class:`~isaaclab_physx.scene_data_providers.PhysxSceneDataProvider`
     - :class:`~isaaclab_newton.scene_data_providers.NewtonSceneDataProvider`
   * - Cloner
     - :func:`~isaaclab.cloner.clone_from_template`
     - :func:`~isaaclab_physx.cloner.physx_replicate`
     - :func:`~isaaclab_newton.cloner.newton_physics_replicate`

The Factory Pattern
-------------------

All factories inherit from :class:`~isaaclab.utils.backend_utils.FactoryBase`, which uses a
**convention-over-configuration** approach to locate backend implementations:

1. The active physics backend is determined by inspecting
   ``SimulationContext.physics_manager``.
2. The factory's module path is used to derive the backend module path by replacing ``isaaclab``
   with ``isaaclab_{backend}``. For example, ``isaaclab.assets.articulation`` maps to
   ``isaaclab_physx.assets.articulation`` or ``isaaclab_newton.assets.articulation``.
3. The backend module is lazily imported and the implementation class is cached in a registry.

.. code-block:: text

    User code: Articulation(cfg)
        │
        ▼
    FactoryBase.__new__()
        │
        ├─ _get_backend()       → "physx" or "newton"
        │    (reads SimulationContext.physics_manager)
        │
        ├─ _get_module_name()   → "isaaclab_physx.assets.articulation"
        │    (convention: isaaclab.X.Y → isaaclab_{backend}.X.Y)
        │
        ├─ importlib.import_module()
        │    (lazy load — only on first use)
        │
        └─ Return backend-specific instance

**Custom backend resolution:** Some factories override the default resolution. For example, the
:class:`~isaaclab.renderers.Renderer` factory selects backends based on the renderer config type
rather than the physics manager, because renderers and physics backends are independent:

.. code-block:: python

    class Renderer(FactoryBase, BaseRenderer):
        _backend_class_names = {
            "physx": "IsaacRtxRenderer",
            "newton": "NewtonWarpRenderer",
            "ov": "OVRTXRenderer",
        }

Similarly, visualizers select backends based on the ``visualizer_type`` field in their config,
allowing any visualizer to work with any physics backend.

Backend Selection
-----------------

The physics backend is selected via the ``physics`` field in
:class:`~isaaclab.sim.SimulationCfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

    # Use PhysX (default)
    sim_cfg = SimulationCfg(physics=PhysxCfg())

    # Use Newton with MuJoCo-Warp solver
    sim_cfg = SimulationCfg(physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(),
        num_substeps=4,
    ))

Once the :class:`~isaaclab.sim.SimulationContext` is initialized, all subsequent factory
instantiations automatically use the selected backend.

Multi-Backend Environments with Presets
---------------------------------------

Environments can support multiple backends simultaneously using the :doc:`preset system
</source/features/hydra>`. Each backend gets its own configuration variant:

.. code-block:: python

    from isaaclab.utils import configclass
    from isaaclab_tasks.utils import PresetCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

    @configclass
    class CartpolePhysicsCfg(PresetCfg):
        default: PhysxCfg = PhysxCfg()
        physx: PhysxCfg = PhysxCfg()
        newton: NewtonCfg = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3)
        )

    @configclass
    class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
        sim: SimulationCfg = SimulationCfg(physics=CartpolePhysicsCfg())

Users then select a backend at the command line:

.. code-block:: bash

    # Default (PhysX)
    python train.py --task Isaac-Cartpole-v0

    # Newton
    python train.py --task Isaac-Cartpole-v0 presets=newton

The Physics Manager
-------------------

Each backend implements :class:`~isaaclab.physics.PhysicsManager`, the abstract base class
that drives the simulation loop:

.. code-block:: python

    class PhysicsManager(ABC):
        @classmethod
        @abstractmethod
        def initialize(cls, sim_context: SimulationContext) -> None: ...

        @classmethod
        @abstractmethod
        def reset(cls, soft: bool = False) -> None: ...

        @classmethod
        @abstractmethod
        def forward(cls) -> None: ...

        @classmethod
        @abstractmethod
        def step(cls) -> None: ...

        @classmethod
        def close(cls) -> None: ...  # concrete; dispatches STOP event

The physics manager also provides a **callback system** via
:class:`~isaaclab.physics.PhysicsEvent` for cross-backend event handling:

.. code-block:: python

    from isaaclab.physics import PhysicsManager, PhysicsEvent

    handle = PhysicsManager.register_callback(
        callback=my_setup_fn,
        event=PhysicsEvent.PHYSICS_READY,
        order=0,
        name="my_callback",
    )

Available events: ``MODEL_INIT`` (during scene building), ``PHYSICS_READY`` (after physics
initialization), and ``STOP`` (on simulation shutdown).

Asset and Sensor Interfaces
---------------------------

Assets and sensors follow the same pattern. Each has:

1. **A base class** in ``isaaclab`` defining the interface (e.g., ``BaseArticulation``,
   ``BaseContactSensor``)
2. **A factory class** that inherits from both ``FactoryBase`` and the base class
3. **Backend implementations** in ``isaaclab_physx`` and ``isaaclab_newton``

The base classes define the public API contract — properties, methods, and data accessors
that all backends must provide. Both backends use ``wp.array`` (Warp arrays) as their
primary data type for asset and sensor data.

Data classes follow the same pattern with their own factories (e.g.,
``ArticulationData(FactoryBase, BaseArticulationData)``).

Adding a New Physics Backend
----------------------------

To add a new physics backend (e.g., ``mybackend``), create a new extension package following
the established conventions:

**1. Package structure:**

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
        │   └── rigid_object_collection/
        │       └── ...
        ├── sensors/
        │   ├── contact_sensor/
        │   └── ...
        ├── renderers/
        │   └── ...
        ├── cloner/
        │   └── ...
        └── scene_data_providers/
            └── ...

**2. Implement the physics manager:**

.. code-block:: python

    # isaaclab_mybackend/physics/mybackend_manager.py
    from isaaclab.physics import PhysicsManager

    class MyBackendManager(PhysicsManager):
        @classmethod
        def initialize(cls, sim_context):
            super().initialize(sim_context)
            # Initialize your physics engine

        @classmethod
        def step(cls):
            # Advance simulation by one timestep

        @classmethod
        def forward(cls):
            # Update kinematics without stepping

        @classmethod
        def reset(cls, soft=False):
            if not soft:
                cls.dispatch_event(PhysicsEvent.PHYSICS_READY)
            # Reset simulation state

        @classmethod
        def close(cls):
            super().close()
            # Clean up resources

**3. Create the physics config:**

.. code-block:: python

    # isaaclab_mybackend/physics/mybackend_manager_cfg.py
    from isaaclab.physics import PhysicsCfg
    from isaaclab.utils import configclass

    @configclass
    class MyBackendCfg(PhysicsCfg):
        class_type = "{DIR}.mybackend_manager:MyBackendManager"
        # Backend-specific settings here

**4. Implement assets and sensors:**

Each asset/sensor must extend the corresponding base class from ``isaaclab``. The class name
must match the factory's expected name (by convention, the same name as the factory class).
Use ``lazy_export()`` in ``__init__.py`` files — no manual registration needed.

.. code-block:: python

    # isaaclab_mybackend/assets/articulation/articulation.py
    from isaaclab.assets.articulation import BaseArticulation

    class Articulation(BaseArticulation):
        def __init__(self, cfg):
            super().__init__(cfg)
            # Set up backend-specific simulation structures

**5. Module discovery is automatic.** The ``FactoryBase`` convention maps
``isaaclab.assets.articulation`` to ``isaaclab_mybackend.assets.articulation`` based on the
active physics manager name. As long as you follow the package structure above, your backend
classes will be discovered automatically.

Key Design Principles
---------------------

- **Lazy loading**: Backend modules are imported only when first instantiated, keeping startup
  fast and avoiding hard dependencies on unused backends.
- **Convention over configuration**: Module paths follow a strict pattern
  (``isaaclab.X.Y`` → ``isaaclab_{backend}.X.Y``), so no manual registration is needed.
- **Independent selection**: Physics backend, renderer, and visualizer are selected
  independently — you can use any combination.
- **Warp-native data types**: Both backends return ``wp.array`` for asset and sensor data.
  Use ``wp.to_torch()`` when interoperating with PyTorch-based code.
- **Zero runtime overhead**: Backend selection happens at instantiation time. There are no
  if-statements or dispatch logic on the hot path.

See Also
--------

- :doc:`/source/migration/migrating_to_isaaclab_3-0` — migration guide from Isaac Lab 2.x to the
  multi-backend architecture
- :doc:`/source/features/hydra` — preset system for multi-backend environment configurations
- :doc:`/source/experimental-features/newton-physics-integration/index` — Newton physics integration
  guide
- :doc:`renderers` — renderer backend architecture
- :doc:`scene_data_providers` — scene data provider architecture
- :doc:`/source/features/visualization` — visualizer backends for interactive feedback
- :doc:`/source/how-to/cloning` — template-based environment cloning guide
