OvPhysX Backend
===============

.. warning::

    OvPhysX is **highly experimental** and is not recommended for general use yet.
    The public surface is changing rapidly while the backend is under active
    development. Expect feature coverage and test commands to change between
    Isaac Lab 3.0 beta releases.

.. warning::

    Do not combine OvPhysX with the Kit visualizer. Commands such as
    ``physics=ovphysx --visualizer kit`` are unsupported because ovphysx
    loads USD-dependent PhysX plugins from its own package, while Kit already
    owns a separate USD/plugin stack in the same process. Use
    ``--visualizer newton``, ``--visualizer rerun``, ``--visualizer viser``,
    or omit ``--visualizer`` for headless execution.

OvPhysX is a kit-less variant of the PhysX backend. It drives PhysX directly
(without the Omniverse Kit runtime) and reads scene-level solver parameters
from the USD ``PhysicsScene`` prim rather than from a Python config. The Python
config :class:`~isaaclab_ov.physics.OvPhysxCfg` only exposes the handful of
GPU buffer sizes that are not represented on the USD schema.

OvPhysX is selected through :class:`~isaaclab_ov.physics.OvPhysxCfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_ov.physics import OvPhysxCfg

    sim_cfg = SimulationCfg(physics=OvPhysxCfg())

Why use OvPhysX?
----------------

* **Kit-less execution.** OvPhysX avoids Omniverse Kit, which makes it a useful
  experimental path for headless deployments and for backends that don't need
  the Kit runtime stack.
* **USD-as-source-of-truth.** Solver parameters are taken from the
  ``PhysicsScene`` USD prim, so authoring tools that already manage USD scenes
  do not need a parallel Python config.

What works today
----------------

The asset and sensor surface tracks PhysX, but only a subset is implemented and
validated at the time of writing. The following pieces are available on
``develop``:

* RigidObject — merged via
  `PR #5426 <https://github.com/isaac-sim/IsaacLab/pull/5426>`_.
* Articulation — merged via
  `PR #5459 <https://github.com/isaac-sim/IsaacLab/pull/5459>`_.
* RigidObjectCollection — merged via
  `PR #5570 <https://github.com/isaac-sim/IsaacLab/pull/5570>`_.
* Contact Sensor — merged via
  `PR #5422 <https://github.com/isaac-sim/IsaacLab/pull/5422>`_.
* SceneDataProvider — merged via
  `PR #5589 <https://github.com/isaac-sim/IsaacLab/pull/5589>`_.
* FrameView — merged via
  `PR #5678 <https://github.com/isaac-sim/IsaacLab/pull/5678>`_.
* :class:`~isaaclab.assets.DeformableObject` — experimental volume- and
  surface-deformable support on CUDA simulation devices.
* Fast-path cloning of heterogeneous rigid-body and articulation geometry
  variants with matching native tensor layouts.

Additional OvPhysX work remains in flight. IMU, Frame Transformer, Joint Wrench,
PVA, Ray Caster, and rendering support are not documented as supported here
until their implementations land on ``develop`` and pass the backend smoke
tests.

Heterogeneous cloning
---------------------

OvPhysX 0.6.3 is required for heterogeneous runtime cloning. Tensor bindings use
numeric environment order so indexed resets, actions and observations address
the correct variant. Variants may differ in geometry but must preserve body,
joint and tendon layout. Isaac Lab checks prototype rotation axes and tendon layouts;
OVPhysX checks body and joint metadata when binding articulations.

The clone context compiles the plan's prototypes and world assignments once.
The physics manager attaches one exported stage, replays the native copies, and
warms the runtime before assets and sensors bind. Hard reset reuses the same
declarations; there is no separate consumable queue. Binding paths come from the
plan's world layout and environment template, not from native clone order.

Configured scenes author only prototypes before replication. Environment poses come from the clone plan:
USD replication creates frames for Kit, and OVPhysX places originals in its private import before native cloning.
OVRTX and OVStage populate their required environment frames in private USD exports, then apply the planned poses.
The SDK's CPU hierarchy computation requires USD-populated frames. Export filtering still excludes unrouted
prototypes and any destinations already materialized by USD replication.

Authored sources all receive native environment ID zero. Every world containing
a source or USD-only physics therefore imports all its declared assets as originals. Other worlds
clone their assets from a complete original world of the same composition,
using one native environment ID per destination world. This keeps a robot,
table and object able to contact one another without destination placeholders.
Only declared asset subtrees are copied; unrelated authored assets are retained.

Keep :attr:`~isaaclab.scene.InteractiveSceneCfg.filter_collisions` enabled to
isolate original worlds with USD collision groups. Native GPU physics copies additionally use
native environment-ID filtering. CPU has no native environment-ID filtering in OvPhysX 0.6.3,
so it imports all declared USD copies and uses collision groups.
Large overlapping original layouts can still exhaust broadphase pair capacity;
use spatially separated environments.

Deformable limitations
----------------------

OvPhysX deformables currently require every body matched by one
:class:`~isaaclab.assets.DeformableObject` to have the same number of simulation
nodes. Initialization raises an error for mixed node counts instead of exposing
padded state that would produce incorrect reductions.

Deformable scenes also require full-stage materialization. Startup cost therefore
grows with the number of authored environments. Use this path for small validation
scenes; training-scale workloads with thousands of environments are not currently
supported. Deformable bodies are not supported by the heterogeneous fast-path
cloner.

Installation
------------

The ``ovphysx`` extra requires OvPhysX 0.6.3. Install it from the repository root with:

.. code-block:: bash

    uv sync --inexact --extra ovphysx

The ``--inexact`` flag preserves packages installed through other extras.
Use ``--extra ov`` to install both public OvPhysX and OVRTX runtimes. The combined
extra pairs OVRTX 0.5.0.377615 with OVStage 0.2; OVRTX 0.4.1 is not compatible
with this runtime combination.

Testing the Installation
------------------------

First check that the Python package and runtime wheel import correctly:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run --extra ovphysx --extra test python -c "import ovphysx.types; from isaaclab_ov.physics import OvPhysxCfg; print('OvPhysX runtime OK')"

Then run a small backend smoke test:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run --extra ovphysx --extra test python -m pytest source/isaaclab_ov/test/assets/test_rigid_object.py::test_initialization -k cpu

To try a task that declares an OvPhysX physics preset, use the same preset CLI
syntax as the other backends:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run --extra ovphysx isaaclab zero_agent --task Isaac-Cartpole-Direct \
              --num_envs 128 --max_steps 64 physics=ovphysx

This command runs a 64-step headless zero-action rollout and then exits.

Supported locomotion environments
---------------------------------

The following locomotion training environments declare an ``ovphysx`` physics
preset. Playback through ``uv run isaaclab play`` uses the same backend where
available.

* ``Isaac-Ant-Direct``
* ``Isaac-Humanoid-Direct``
* ``IsaacContrib-Velocity-Rough-AnymalB``
* ``IsaacContrib-Velocity-Rough-AnymalC``
* ``Isaac-Velocity-Flat-AnymalD``
* ``Isaac-Velocity-Rough-AnymalD``
* ``IsaacContrib-Velocity-Rough-UnitreeA1``
* ``IsaacContrib-Velocity-Rough-UnitreeGo1``
* ``Isaac-Velocity-Rough-UnitreeGo2``
* ``Isaac-Velocity-Rough-Cassie``
* ``Isaac-Velocity-Rough-G1``
* ``Isaac-Velocity-Rough-H1``

Status and follow-up
--------------------

OvPhysX is still experimental, so the feature list above is intentionally
conservative. Broader feature coverage and documentation parity are tracked in
`issue #5634 <https://github.com/isaac-sim/IsaacLab/issues/5634>`_.

For architectural context, see :ref:`backend-architecture`.

For raw ``TensorBinding`` access and :class:`~isaaclab_ov.sim.views.OvPhysxView`,
see :doc:`/source/concepts/native-physics-api/ovphysx`.
