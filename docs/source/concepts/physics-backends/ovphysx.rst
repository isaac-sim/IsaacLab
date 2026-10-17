.. _ovphysx-backend:

OvPhysX
========

.. warning::

    Do not combine OvPhysX with the Kit visualizer. Commands such as
    ``physics=ovphysx --visualizer kit`` are unsupported because ovphysx
    loads USD-dependent PhysX plugins from its own package, while Kit already
    owns a separate USD/plugin stack in the same process. Use
    ``--visualizer newton_gl``, ``--visualizer rerun``, ``--visualizer viser``,
    or omit ``--visualizer`` for headless execution.

OvPhysX runs PhysX without Omniverse Kit. It reads scene-level solver settings
from the USD ``PhysicsScene`` prim. :class:`~isaaclab_ov.physics.OvPhysxCfg` holds
GPU buffer settings that USD does not represent.

OvPhysX is selected through :class:`~isaaclab_ov.physics.OvPhysxCfg`:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_ov.physics import OvPhysxCfg

    sim_cfg = SimulationCfg(physics=OvPhysxCfg())

Why use OvPhysX?
----------------

* **No Kit runtime.** Run headless simulations without Isaac Sim.
* **USD configuration.** Scene-authoring tools can set solver parameters directly
  on the ``PhysicsScene`` prim.

What works today
----------------

OvPhysX supports the following components. The notes describe requirements and
limits to consider when configuring a task.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Coverage and limits
   * - Rigid objects and articulations
     - Rigid objects, rigid-object collections, and articulations.
   * - Contact sensing
     - Aggregate and filtered forces, contact points, and contact timing.
       Pose tracking requires one sensor body per environment.
   * - Motion and frame sensing
     - IMU, PVA, and frame transforms. Joint-wrench readings include the root body
       and use the child-side joint frame.
   * - Ray casting
     - Ray casters and ray-cast cameras using Warp meshes. The standard ray caster
       requires static target meshes.
   * - Rendering and scene data
     - Scene-data and frame-view implementations support kit-less consumers,
       including OVRTX. Isaac Sim / Kit cannot share the process.
   * - Deformables
     - Volume and surface deformables on CUDA. Bodies in one view
       need matching node counts; see `Deformable limitations`_.
   * - Heterogeneous cloning
     - Geometry variants with matching body, joint, and tendon layouts;
       see `Heterogeneous cloning`_.

See :doc:`/source/concepts/sensors/index` for sensor configuration and
:doc:`/source/concepts/renderers` for renderer choices.

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

..
   Implementation history: rigid objects #5426, articulations #5459,
   rigid-object collections #5570, contacts #5422, scene data #5589,
   and frame views #5678 in https://github.com/isaac-sim/IsaacLab/pulls.
   Broader integration work was tracked in issue #5634.
   Check isaaclab_ov implementations and tests when updating this table.

Installation
------------

The ``ovphysx`` extra requires OvPhysX 0.6.4.72118723. Install it from the repository root with:

.. code-block:: bash

    uv sync --inexact --extra ovphysx

The ``--inexact`` flag preserves packages installed through other extras.
Use ``--extra ov`` to install both public OvPhysX and OVRTX runtimes. The combined
extra pairs OVRTX 0.5.1.385782 with OVStage 0.2.1; OVRTX 0.4.1 is not compatible
with this runtime combination.

Opt-in OVPhysX 0.7 support
~~~~~~~~~~~~~~~~~~~~~~~~~~

The default dependency pins remain on OVPhysX 0.6.4. The backend also accepts
OVPhysX 0.7, including development builds. It selects task completion, stage
teardown, scene reads and schema discovery from the installed runtime version.
Tensor and contact bindings remain in use on both versions.

OVPhysX 0.7 no longer bundles PhysX USD schemas. Install ``physx-usd-schemas>=25.11.2``
into the simulation environment before startup. Use ``uv run --no-sync`` with an
opt-in environment to avoid restoring the project's default dependency pins.
Isaac Lab locates
both PhysX and deformable schema directories without importing the provider,
then registers them before USD population. The already-installed Newton schemas
are also registered. Do not suppress schema-registration errors: late registration
requires a fresh process and cannot recover settings lost during population.

Legacy joint USD
~~~~~~~~~~~~~~~~

OVPhysX 0.7 no longer consumes ``PhysxJointAPI``. In the temporary physics export,
Isaac Lab translates authored ``physxJoint:maxJointVelocity`` and
``physxJoint:armature`` to ``NewtonJointAPI`` values. Angular velocity stays in
USD degrees per second; linear velocity and armature retain their USD units.
Authored modern values take precedence, and existing per-axis PhysX overrides
remain intact. Source assets and the live stage are not edited, so other physics
backends and older OVPhysX runtimes keep their previous behavior.

Unmigrated nonzero legacy ``physxJoint:jointFriction`` is rejected with the joint path.
The dimensionless coefficient has no exact conversion to the new effort-based
friction model. Remove it and retune ``newton:friction`` / ``newton:damping`` or
``PhysxJointAxisAPI`` friction efforts explicitly. Unsupported joint types and
time-sampled legacy settings that cannot be represented also require migration.
This compatibility layer does not promise identical trajectories across PhysX
engine versions.

For scene-time migration without republishing assets, set
``OvPhysxCfg.legacy_joint_friction_replacements`` to a mapping from joint-name
regular expressions to ``(static effort, dynamic effort, viscous coefficient)``.
The values use SI force/torque and viscous units. Only matching revolute and
prismatic joints with nonzero legacy friction are migrated in the private 0.7
export: the legacy property is removed and modern ``PhysxJointAxisAPI`` values
are authored. Existing modern friction opinions take precedence. Other runtimes,
source USD files, and the live stage remain unchanged.

The Allegro reorientation, Kuka-Allegro lift/reorientation, Shadow reorientation,
and Shadow handover presets supply replacements for their affected hand joints.
The profile is ``(0.01, 0.0, 0.0)``: 0.01 N*m static effort, zero dynamic effort,
and zero viscous coefficient. These values match measured OVPhysX 0.6.4 runtime
actuator settings, including the Shadow tendon joints; they are not a numeric
conversion of the legacy asset coefficients. Actuator initialization and runtime
friction randomization continue using effort-valued tensor properties. Merely
copying a legacy coefficient into a modern effort property is not a valid migration.

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

          uv run --extra ovphysx --extra test python -m pytest source/isaaclab_ov/test/assets/test_rigid_object.py::test_rigid_object_real_ovphysx_seams -k cpu

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

Use the :doc:`environment browser </source/isaaclab/environments>` or task help
for the available ``physics=`` presets. To list launchable combinations:

.. code-block:: bash

   uv run --extra ovphysx isaaclab list_envs --show_presets --keyword Velocity

Playback uses the same ``physics=ovphysx`` selector as training.

.. dropdown:: Locomotion task examples

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

Authored sources all receive native environment ID zero. Every world containing
a source or USD-only physics therefore imports all its declared assets as originals. Other worlds
clone their assets from a complete original world of the same composition,
using one native environment ID per destination world. This keeps a robot,
table and object able to contact one another without destination placeholders.
Only declared asset subtrees are copied; unrelated authored assets are retained.

Keep :attr:`~isaaclab.scene.InteractiveSceneCfg.filter_collisions` enabled to
isolate original worlds with USD collision groups. GPU copies additionally use
native environment-ID filtering. CPU has no native environment-ID filtering
in OvPhysX 0.6.3, so it imports all declared USD copies and uses collision groups.
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

.. _ovphysx-native-access:

Native access
-------------

Mental model
~~~~~~~~~~~~

OvPhysX exposes generic ``TensorBinding`` objects. A binding combines a prim
selection with one installed ``TensorType``; it is not an asset-specific native
storage object. Callers explicitly pull values into their own buffer with
``read()`` and push a buffer back with ``write()``. Isaac Lab optionally wraps
these bindings in :class:`~isaaclab_ov.sim.views.OvPhysxView` for a
string-keyed, guarded convenience surface.

Lifecycle prerequisite
~~~~~~~~~~~~~~~~~~~~~~

Create bindings only after the OvPhysX simulation has initialized and reset,
when its native runtime is live. A stage reload, a reset path that rebuilds the
stage or runtime, and manager teardown invalidate all existing bindings and
views. Reacquire them afterwards.

Get the active OvPhysX handle from the manager after the lifecycle prerequisite:

.. code-block:: python

   from isaaclab_ov.physics import OvPhysxManager

   physx = OvPhysxManager.get_physx_instance()
   if physx is None:
       raise RuntimeError("OvPhysX has not been constructed; initialize and reset the simulation first.")

A non-``None`` handle proves only that the manager constructed an OvPhysX instance; it is not a
readiness predicate for the current simulation context. Manager teardown releases the instance
and clears this handle. Initialize and reset the current context before access, and always
reacquire the handle before creating new bindings or views.

Reuse Isaac Lab-owned access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuse the convenience view already owned by an Isaac Lab asset when its
selection matches:

.. code-block:: python

   object_view = scene["object"].root_view
   poses = object_view.get_attribute("rigid_body_pose")

Create raw access
~~~~~~~~~~~~~~~~~

Create a raw binding when the desired selection is not already owned by an
Isaac Lab asset. The following representative pose binding is float32 and
uses the simulation device; it is not a generic allocator for every tensor
type:

.. code-block:: python

   import warp as wp
   from isaaclab.physics import PhysicsManager
   from ovphysx.types import TensorType

   binding = physx.create_tensor_binding(
       pattern="/World/envs/env_*/Object",
       tensor_type=TensorType.RIGID_BODY_POSE,
   )
   try:
       poses = wp.empty(
           tuple(binding.shape),
           dtype=wp.float32,
           device=PhysicsManager.get_device(),
       )
       binding.read(poses)
       binding.write(poses)
   finally:
       binding.destroy()

Before allocating a buffer, inspect ``binding.shape``, ``binding.dtype``, the
binding's count, and its path metadata. Match the binding shape and DLPack scalar
metadata exactly, and respect the binding's native device and access mode.
``RIGID_BODY_POSE`` has the float32 layout used above, but another tensor type
can have a different scalar dtype, shape, native device, or write permission.

For a new selection, construct :class:`~isaaclab_ov.sim.views.OvPhysxView`
with the native runtime and a Tensor API pattern:

.. code-block:: python

   from isaaclab.physics import PhysicsManager
   from isaaclab_ov.sim.views import OvPhysxView

   view = OvPhysxView(
       physx,
       pattern="/World/envs/env_*/Object",
       device=PhysicsManager.get_device(),
   )
   try:
       if view.try_binding_for("rigid_body_pose") is not None:
           poses = view.get_attribute("rigid_body_pose")
           view.set_attribute("rigid_body_pose", poses)
           view.read_into("rigid_body_pose", poses)
   finally:
       view.close()

``try_binding_for`` attempts to create a binding for this selection and returns
``None`` when a valid type is unavailable. ``binding_for`` returns a raw binding
and bypasses the view's device, dtype, shape, and read-only guards.

Read/write semantics
~~~~~~~~~~~~~~~~~~~~

Raw ``binding.read(buffer)`` calls pull values into caller-owned buffers, while
``binding.write(buffer)`` calls push values to the simulation. Editing a local
buffer alone does not change the simulation. The guarded
:meth:`~isaaclab_ov.sim.views.OvPhysxView.get_attribute`,
:meth:`~isaaclab_ov.sim.views.OvPhysxView.read_into`, and
:meth:`~isaaclab_ov.sim.views.OvPhysxView.set_attribute` paths provide the
same pull/push boundary while validating the binding's shape, scalar dtype,
native device, and writable access.

Ownership, synchronization, and invalidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allocate buffers on the binding's required device and match its shape and
DLPack scalar metadata. State tensors normally use the simulation device, while
some property tensors are CPU-only. :class:`~isaaclab_ov.sim.views.OvPhysxView`
validates these requirements and can reinterpret supported flat scalar layouts
as structured Warp values, but it does not move data between CPU and GPU.

Use ``try_binding_for`` when a tensor type might not apply to the selected
prims. Reacquire raw bindings and convenience views after reset paths that
rebuild the stage or runtime, or after teardown. Destroy raw bindings and close
custom views before they are no longer needed. Manager teardown closes tracked
``OvPhysxView`` instances before releasing the runtime, but explicit cleanup
avoids retaining their bindings for the rest of a long-lived simulation.

Authoritative references
~~~~~~~~~~~~~~~~~~~~~~~~

The generated Isaac Lab reference for
:class:`~isaaclab_ov.sim.views.OvPhysxView` documents the convenience
view. The repository metadata pins OvPhysX to a version but does not provide
a versioned official OvPhysX documentation URL, so inspect the installed
``TensorType`` and binding metadata at runtime.

.. _status-and-follow-up:

Where to go next
----------------

* :doc:`/source/isaaclab/environments` — tasks and backend presets.
* :ref:`backend-architecture` — shared factories and resource ownership.
* :doc:`/source/concepts/native-physics-api/index` — native access across backends.
