Multi-Backend Architecture
==========================

.. seealso::

   This page is the source of truth for the ``isaaclab-selecting-backends`` and
   ``isaaclab-using-presets`` agent skills
   (`skills/user/select-backends/ <../../../../skills/user/select-backends/SKILL.md>`__,
   `skills/user/use-presets/ <../../../../skills/user/use-presets/SKILL.md>`__).
   When you change this page, update those skills so agent guidance stays in sync. See
   :doc:`/source/overview/developer-guide/agent_skills`.

Isaac Lab 3.0 introduced a multi-backend architecture that enables running simulations with
different physics backends (PhysX, Newton, and OvPhysX) while maintaining a unified API.
This page explains how the backend system works and how to extend it.

Overview
--------

Instead of hard-coding a single physics engine, Isaac Lab uses a **factory pattern** to
dispatch object creation to backend-specific implementations at runtime. When you write:

.. code-block:: python

    from isaaclab.assets import Articulation

    robot = Articulation(cfg)

The ``Articulation`` class is a factory that automatically creates an instance of
the active backend implementation, such as
:class:`PhysX Articulation <isaaclab_physx.assets.Articulation>`,
:class:`Newton Articulation <isaaclab_newton.assets.Articulation>`, or
:class:`OvPhysX Articulation <isaaclab_ov.assets.Articulation>`. Your code never
needs to import backend-specific modules directly.

This pattern applies across simulation components, though not every backend implements every
component yet:

.. list-table::
   :header-rows: 1

   * - Component
     - Core API (``isaaclab``)
     - PhysX (``isaaclab_physx``)
     - Newton (``isaaclab_newton``)
     - OvPhysX (``isaaclab_ov``)
   * - Physics Manager
     - :class:`~isaaclab.physics.PhysicsManager`
     - :class:`~isaaclab_physx.physics.PhysxManager`
     - :class:`~isaaclab_newton.physics.NewtonManager`
     - :class:`~isaaclab_ov.physics.OvPhysxManager`
   * - Articulation
     - :class:`~isaaclab.assets.Articulation`
     - :class:`~isaaclab_physx.assets.Articulation`
     - :class:`~isaaclab_newton.assets.Articulation`
     - :class:`~isaaclab_ov.assets.Articulation`
   * - Rigid Object
     - :class:`~isaaclab.assets.RigidObject`
     - :class:`~isaaclab_physx.assets.RigidObject`
     - :class:`~isaaclab_newton.assets.RigidObject`
     - :class:`~isaaclab_ov.assets.RigidObject`
   * - Deformable Object
     - :class:`~isaaclab.assets.DeformableObject`
     - :class:`~isaaclab_physx.assets.DeformableObject`
     - :class:`~isaaclab_newton.assets.DeformableObject`
     - Not supported
   * - Cable Object
     - :class:`~isaaclab.assets.CableObject`
     - Not supported
     - :class:`~isaaclab_newton.assets.CableObject`
     - Not supported
   * - Contact Sensor
     - :class:`~isaaclab.sensors.ContactSensor`
     - :class:`~isaaclab_physx.sensors.ContactSensor`
     - :class:`~isaaclab_newton.sensors.ContactSensor`
     - :class:`~isaaclab_ov.sensors.ContactSensor`
   * - Renderer
     - :class:`~isaaclab.renderers.BaseRenderer`
     - :class:`~isaaclab_physx.renderers.IsaacRtxRenderer`
     - :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`
     - Not supported
   * - Scene Data Backend
     - :class:`~isaaclab.scene_data.SceneDataBackend`
     - ``PhysxSceneDataBackend`` (in :mod:`isaaclab_physx.physics`)
     - ``NewtonSceneDataBackend`` (in :mod:`isaaclab_newton.physics`)
     - ``OvPhysxSceneDataBackend`` (in :mod:`isaaclab_ov.physics`)
   * - Cloner
     - :class:`~isaaclab.cloner.UsdReplicateContext`
     - :class:`~isaaclab_physx.cloner.PhysxReplicateContext`
     - :class:`~isaaclab_newton.cloner.NewtonReplicateContext`
     - :class:`~isaaclab_ov.cloner.OvPhysxReplicateContext`

Each context consumes the same :class:`~isaaclab.cloner.ClonePlan` through
``context.replicate(plan)``. The core cloner owns plan construction and dispatch;
backend contexts only execute the rows routed to them.

The Factory Pattern
-------------------

All factories inherit from :class:`~isaaclab.utils.backend_utils.FactoryBase`, which uses a
**convention-over-configuration** approach to locate backend implementations:

1. The active physics backend is determined by inspecting
   ``SimulationContext.physics_manager``.
2. The factory's module path is used to derive the backend module path. For example,
   ``isaaclab.assets.articulation`` maps to ``isaaclab_physx.assets.articulation``,
   ``isaaclab_newton.assets.articulation``, or ``isaaclab_ov.assets.articulation``.
   The OVPhysX backend key maps to the shared ``isaaclab_ov`` integration package.
3. The backend module is lazily imported and the implementation class is cached in a registry.

.. code-block:: text

    User code: Articulation(cfg)
        │
        ▼
    FactoryBase.__new__()
        │
        ├─ _get_backend()       → "physx", "newton", or "ovphysx"
        │    (reads SimulationContext.physics_manager)
        │
        ├─ _get_module_name()   → "isaaclab_physx.assets.articulation"
        │    (OVPhysX maps to the shared isaaclab_ov package)
        │
        ├─ importlib.import_module()
        │    (lazy load — only on first use)
        │
        └─ Return backend-specific instance

Renderers and visualizers do not use this physics factory. Their concrete configs own the
implementation class, and their composition roots construct it with the same convention used by
other declarative configs:

.. code-block:: python

    renderer = renderer_cfg.class_type(renderer_cfg)
    visualizer = visualizer_cfg.class_type(visualizer_cfg)

The config determines the implementation independently of the physics backend. ``RenderContext``
retains one renderer for each equal renderer config, while ``SimulationContext`` owns visualizer
construction and initialization.

Backend Selection
-----------------

The physics backend is selected via the ``physics`` field in
:class:`~isaaclab.sim.SimulationCfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg

    # Use PhysX (default)
    sim_cfg = SimulationCfg(physics=PhysxCfg())

    # Use Newton with MuJoCo-Warp solver
    sim_cfg = SimulationCfg(physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(),
        num_substeps=4,
    ))

    # Use OvPhysX
    sim_cfg = SimulationCfg(physics=OvPhysxCfg())

Once the :class:`~isaaclab.sim.SimulationContext` is initialized, all subsequent factory
instantiations automatically use the selected backend.

Multi-Backend Environments with Presets
---------------------------------------

Environments can support multiple backends simultaneously using :doc:`backend and preset
selectors </source/concepts/backends_and_presets>`. Each backend gets its own configuration
variant. The example below shows only the physics-related fields:

.. code-block:: python

    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.physics import PhysxAutoCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_tasks.utils import PresetCfg

    @configclass
    class CartpolePhysicsCfg(PresetCfg):
        isaacsim_physx: PhysxCfg = PhysxCfg()
        ovphysx: OvPhysxCfg = OvPhysxCfg()
        physx: PhysxAutoCfg = PhysxAutoCfg(
            isaacsim_physx=isaacsim_physx,
            ovphysx=ovphysx,
        )
        default: PhysxCfg = isaacsim_physx
        newton_mjwarp: NewtonCfg = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3)
        )

    @configclass
    class CartpoleEnvCfg(DirectRLEnvCfg):
        sim: SimulationCfg = SimulationCfg(physics=CartpolePhysicsCfg())

Users then select a physics backend at the command line:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          # Default (concrete Isaac Sim PhysX)
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct

          # Automatic PhysX-family selection
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=physx

          # MJWarp (Newton backend)
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=newton_mjwarp

          # OvPhysX backend
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=ovphysx

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          # Default (concrete Isaac Sim PhysX)
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct

          # Automatic PhysX-family selection
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=physx

          # MJWarp (Newton backend)
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=newton_mjwarp

          # OvPhysX backend
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=ovphysx

When a task's default would otherwise be automatic ``PhysxAutoCfg`` selection,
its ``default`` variant is the concrete ``isaacsim_physx`` configuration.
Explicit defaults such as Newton remain unchanged. The ``physics=physx``
selector is opt-in and chooses between Isaac Sim PhysX and OvPhysX at launch
time according to whether the resolved runtime requires Kit. This mirrors
renderer presets: the default is concrete ``isaacsim_rtx``, while
``renderer=rtx`` opts into automatic selection.

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
3. **Backend implementations** in ``isaaclab_physx``, ``isaaclab_newton``, and
   ``isaaclab_ov`` where supported

The base classes define the public API contract — properties, methods, and data accessors
that all backends must provide. Current backend implementations use ``wp.array``
(Warp arrays) as their primary data type for asset and sensor data.

Data classes follow the same pattern with their own factories (e.g.,
``ArticulationData(FactoryBase, BaseArticulationData)``).

These base interfaces define the portable contract. Advanced code can also use
each engine's native low-level data API, but those APIs deliberately retain
different ownership and synchronization semantics. See
:doc:`physical-backends/direct-api-access/index` for PhysX typed views, Newton
live model/state arrays and generic selections, and OvPhysX tensor bindings.

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

**2. Implement the physics manager:**

The manager must expose a :class:`~isaaclab.scene_data.SceneDataBackend` so that
:class:`~isaaclab.scene_data.SceneDataProvider` can read your backend's body
transforms in a Warp-native format that renderers and visualizers consume directly.

.. code-block:: python

    # isaaclab_mybackend/physics/mybackend_manager.py
    from isaaclab.physics import PhysicsManager
    from isaaclab.scene_data import SceneDataBackend, SceneDataFormat


    class MyBackendSceneDataBackend(SceneDataBackend):
        def __init__(self):
            self._scene_data = SceneDataFormat.Transform()

        @property
        def transforms(self) -> SceneDataFormat.Transform:
            # Return current world-space body transforms as a Warp ``transformf`` array.
            self._scene_data.transforms = ...  # backend-native tensor view
            return self._scene_data

        @property
        def transform_count(self) -> int:
            ...

        @property
        def transform_paths(self) -> list[str]:
            # Prim path per row of ``transforms``; used by ``SceneDataProvider.create_mapping``.
            ...


    class MyBackendManager(PhysicsManager):
        _scene_data_backend: ClassVar[MyBackendSceneDataBackend | None] = None

        @classmethod
        def initialize(cls, sim_context):
            super().initialize(sim_context)
            cls._scene_data_backend = MyBackendSceneDataBackend()
            # Initialize your physics engine

        @classmethod
        def get_scene_data_backend(cls) -> SceneDataBackend:
            return cls._scene_data_backend

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
    from isaaclab.utils.configclass import configclass

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
- **Convention over configuration**: Backend module paths mirror the ``isaaclab.X.Y``
  structure. OVPhysX maps to ``isaaclab_ov.X.Y``; other backends use their
  ``isaaclab_<backend>.X.Y`` package, so no manual registration is needed.
- **Independent selection**: Physics backend, renderer, and visualizer are selected
  independently — you can use any combination.
- **Warp-native data types**: Backend implementations return ``wp.array`` for asset and
  sensor data. Use ``wp.to_torch()`` when interoperating with PyTorch-based code.
- **Zero runtime overhead**: Backend selection happens at instantiation time. There are no
  if-statements or dispatch logic on the hot path.

See Also
--------

- :doc:`/source/migration/migrating_to_isaaclab_3-0` — migration guide from Isaac Lab 2.x to the
  multi-backend architecture
- :doc:`/source/concepts/backends_and_presets` — user guide to backend and preset selection
- :doc:`/source/features/hydra` — advanced configuration and preset authoring
- :doc:`physical-backends/index` — feature matrix and per-backend guides (PhysX, Newton, OvPhysX)
- :doc:`physical-backends/newton/index` — Newton backend guide
- :doc:`physical-backends/newton/newton-manager-abstraction` — adding Newton solver managers and
  coupled solvers
- :doc:`renderers` — renderer backend architecture
