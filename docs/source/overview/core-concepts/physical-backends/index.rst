Physics Backends
================

Isaac Lab 3.0 supports multiple physics backends through a unified API. Each backend
exposes the same :class:`~isaaclab.assets.Articulation`,
:class:`~isaaclab.assets.RigidObject`, sensor and renderer surfaces, while differing
in solver characteristics, maturity, and feature coverage. See
:doc:`../multi_backend_architecture` for how the dispatch and factory machinery work
under the hood.

This page summarizes what each backend supports today; the sub-pages document
backend-specific configuration, installation, and limitations.

.. toctree::
    :maxdepth: 2

    physx/index
    newton/index
    ovphysx/index
    solver-comparison


Choosing a Backend
------------------

* **PhysX** — the historical default. Production-ready, broad coverage of Isaac Lab
  features, and the reference for behavior parity. Selected via
  :class:`~isaaclab_physx.physics.PhysxCfg`.
* **Newton** — GPU-accelerated, Warp-native, and differentiable. The Newton
  integration ships with the MuJoCo-Warp solver and beta support for the Kamino
  solver. Selected via :class:`~isaaclab_newton.physics.NewtonCfg`.
* **OvPhysX** — a **highly experimental** kit-less PhysX backend that reads
  scene-level parameters from the USD ``PhysicsScene`` prim. Selected via
  :class:`~isaaclab_ovphysx.physics.OvPhysxCfg`. Not recommended for general use yet.

The active backend is selected at simulation construction time and applies to every
asset, sensor, and renderer instantiated thereafter:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

    # PhysX (default)
    sim_cfg = SimulationCfg(physics=PhysxCfg())

    # Newton with MuJoCo-Warp
    sim_cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))


Feature Support Matrix
----------------------

The matrix below is intentionally coarse-grained. For exhaustive per-asset and
per-task support, see each backend's own ``limitations`` page.

.. list-table::
    :header-rows: 1
    :widths: 30 20 30 20

    * - Feature
      - PhysX
      - Newton
      - OvPhysX
    * - Maturity
      - Stable
      - Beta
      - Highly experimental
    * - Default solver
      - TGS (rigid body)
      - MuJoCo-Warp
      - PhysX (TGS / PGS via USD)
    * - Alternative solvers
      - PGS
      - Kamino (beta), additional Newton solvers planned
      - —
    * - Differentiable
      - No
      - Yes (via Warp)
      - No
    * - Articulation API
      - Yes
      - Yes
      - In-flight (PR #5459)
    * - Rigid Object API
      - Yes
      - Yes
      - Yes
    * - Contact Sensor
      - Yes
      - Yes
      - In-flight (PR #5422)
    * - IMU
      - Yes
      - Yes
      - In-flight (PR #5421)
    * - Frame Transformer / Ray Caster / PVA / Joint-Wrench Sensor
      - Yes
      - Yes
      - Not yet
    * - Camera / Tiled Rendering
      - Yes (RTX)
      - Yes (Newton-Warp renderer)
      - Not yet
    * - Requires Isaac Sim
      - Yes
      - Optional (only for the Omniverse visualizer)
      - Yes
    * - Solver configuration source
      - :class:`~isaaclab_physx.physics.PhysxCfg`
      - :class:`~isaaclab_newton.physics.NewtonCfg` + solver config
      - USD ``PhysicsScene`` + :class:`~isaaclab_ovphysx.physics.OvPhysxCfg`


Selecting Backends per Task
---------------------------

Tasks that support more than one backend define a Hydra preset on
``SimulationCfg.physics``. The example below shows the cartpole task config which
declares all three backends side by side:

.. code-block:: python

    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    from isaaclab_ovphysx.physics import OvPhysxCfg

    @configclass
    class CartpolePhysicsCfg(PresetCfg):
        default: PhysxCfg = PhysxCfg()
        physx: PhysxCfg = PhysxCfg()
        newton_mjwarp: NewtonCfg = NewtonCfg(solver_cfg=MJWarpSolverCfg())
        ovphysx: OvPhysxCfg = OvPhysxCfg()

Users then select the backend at the command line via ``presets=<name>`` or by
overriding the physics field directly. See :ref:`hydra-backend-solver-presets` for
the full Hydra interaction.
