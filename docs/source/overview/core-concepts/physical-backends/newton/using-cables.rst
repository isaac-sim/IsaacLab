.. _newton-using-cables:

Using Cables
============

Isaac Lab exposes 1D cable / rod assets on the Newton backend. A cable is
authored as a single open ``UsdGeom.BasisCurves`` prim carrying the
``PhysicsCurvesDeformableSimAPI`` schema, and is simulated by Newton's VBD
solver as a chain of per-segment capsule bodies joined by ``JointType.CABLE``
joints, with thickness, density, stretch stiffness, and bend stiffness.

Cable support is experimental. The spawner cfg, asset class, and material
defaults may change while Newton cable support is under active development.

.. note::
    Cables are **only supported on the Newton backend**, and only under its VBD
    solver. :class:`~isaaclab.assets.CableObject` is a backend-dispatched
    factory: selecting PhysX or OpenUSD PhysX raises the factory import error at
    construction, so a misconfigured scene fails fast instead of loading the
    curve as inert geometry.


Quick Start: The Cable Demo
---------------------------

Before adding cables to a task, run the standalone demo to confirm that the
spawner, the VBD solver, collision, and the Kit / Fabric viewport sync all work
in your environment:

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/cables.py
    ./isaaclab.sh -p scripts/demos/cables.py --num_cables 40 --num_segments 15

The demo spawns a pile of randomly oriented cables onto a ground plane under
standalone Newton VBD, lets them collide and settle, and periodically restores
them to their spawn state. Source: ``scripts/demos/cables.py``.


Authoring a Cable
-----------------

A cable is configured with a :class:`~isaaclab.sim.spawners.shapes.CableCfg`
plus a :class:`~isaaclab.sim.spawners.materials.CableMaterialCfg`. Adjacent
pairs in ``positions`` become individual segments, each materialized as a
capsule body of diameter :attr:`CableMaterialCfg.thickness` and joined to its
neighbour by a Newton cable joint. ``N`` control points produce ``N-1`` segment
bodies and ``N-2`` cable joints; the root segment is free-floating.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``positions``
      - Control points in the cable-local frame [m]. Requires at least **three**
        finite points, with consecutive points separated by more than ``1e-8`` m.
    * - ``physics_material``
      - Required :class:`~isaaclab.sim.spawners.materials.CableMaterialCfg`; see
        `Cable Material Parameters`_ below. Thickness is also written to the
        curve's ``widths`` attribute so the visual radius matches the physics.
    * - ``collision_props``
      - Optional collision properties. When omitted, the cable is
        collision-free; when set (typically
        ``[UsdPhysicsCollisionCfg(collision_enabled=True)]``), the cable collides
        with the ground and other cables. See `Collision`_ below.
    * - ``visual_material``
      - Optional :class:`~isaaclab.sim.spawners.materials.VisualMaterialCfg` for
        the curve's appearance.
    * - ``visual_material_path``
      - Sub-path of the visual material under the cable geometry prim. Defaults
        to ``"material"``.
    * - ``physics_material_path``
      - Sub-path of the physics material under the cable geometry prim. Defaults
        to ``"physics_material"``.

.. code-block:: python

    import isaaclab.sim as sim_utils

    cable_spawn = sim_utils.CableCfg(
        positions=[(index * 0.1, 0.0, 0.0) for index in range(10)],
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2)),
        physics_material=sim_utils.CableMaterialCfg(
            thickness=0.03,
            density=1000.0,
            stretch_stiffness=1.0e9,
            bend_stiffness=1.0e6,
        ),
        collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
    )

This authors a straight 0.9 m red cable along the cable-local x-axis: 9 capsule
segments (0.1 m long, 0.03 m diameter) joined by 8 cable joints. See
`Loading Cables from USD`_ for how the curve becomes Newton physics.

Wrap the spawner in a :class:`~isaaclab.assets.CableObjectCfg` to get a runtime
asset whose per-segment state can be read, written, and restored:

.. code-block:: python

    from isaaclab.assets import CableObject, CableObjectCfg

    cable = CableObject(
        cfg=CableObjectCfg(
            prim_path="/World/Env_0/Cable",
            spawn=cable_spawn,
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
    )

The inherited ``init_state`` sets the cable root's spawn pose in each
environment.


Picking a Solver
----------------

Cables can **only** be simulated under Newton's VBD solver, which is the only
solver that steps ``JointType.CABLE`` joints. A cable-only scene uses a
standalone :class:`~isaaclab_contrib.deformable.VBDSolverCfg`:

.. code-block:: python

    from isaaclab_newton.physics import NewtonCfg

    from isaaclab_contrib.deformable import VBDSolverCfg

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=20), num_substeps=8),
    )

Mixed rigid + cable scenes (for example a robot manipulating a cable) run the
cable under a :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` entry: define a
named VBD entry that owns the cable segments and couple it to the rigid entry.
See :doc:`using-vbd-solver`.


Cable Material Parameters
-------------------------

:class:`~isaaclab.sim.spawners.materials.CableMaterialCfg` defines the cable's
geometry and stiffness. All values are validated at author time; nonfinite or
out-of-range values raise :class:`ValueError` before any prim is created. The
attributes are authored in the standard ``physics:`` namespace and read back by
Newton's importer.

.. list-table::
    :header-rows: 1
    :widths: 24 76

    * - Parameter
      - Description
    * - ``thickness``
      - Full cable thickness (diameter) [m]. Must be finite and positive.
        Newton uses ``radius = thickness / 2`` for the capsule cross-section,
        the bending second moment of area, and the collision radius. Default
        ``0.001``.
    * - ``density``
      - Cable density [kg/m^3]. Must be finite and positive. Newton derives
        per-segment mass from the density and the capsule volume. Default
        ``1000.0``.
    * - ``stretch_stiffness``
      - Axial (stretch) elastic modulus ``E`` [Pa], i.e. force per area. Must be
        finite and nonnegative. Newton converts it to the rod's per-joint axial
        stiffness ``E * A / L``, where ``A`` is the circular cross-section area
        and ``L`` is the mean segment rest length. Higher values reduce elongation
        but need more solver iterations or substeps. Default ``1.0e9``.
    * - ``bend_stiffness``
      - Bending elastic modulus ``E`` [Pa]. Must be finite and nonnegative.
        Newton converts it to the per-joint bend stiffness ``E * I / L``, where
        ``I`` is the second moment of area of the circular cross-section.
        ``0.0`` gives a limp rope; increase for a stiff hose or wire. Default
        ``1.0e6``.
    * - ``shear_stiffness``
      - Transverse shear elastic modulus [Pa]. Must be finite and nonnegative.
        Optional: when left at ``None`` the attribute is not authored and the solver
        falls back to :attr:`stretch_stiffness`. Default ``None``.
    * - ``twist_stiffness``
      - Torsional elastic modulus [Pa]. Must be finite and nonnegative. Optional:
        when left at ``None`` the attribute is not authored and the solver falls back
        to :attr:`bend_stiffness`. Default ``None``.

.. note::
    A Newton cable joint has four degrees of freedom: linear stretch and shear, and
    angular bend and twist. Leaving :attr:`shear_stiffness` or :attr:`twist_stiffness`
    unset does **not** mean the cable has no shear or twist resistance; it means the
    solver reuses the stretch and bend moduli for them. Set them explicitly to decouple
    torsion from bending, for example a hose that bends easily but resists twisting.
    Authoring ``0.0`` is distinct from leaving them unset: it removes that resistance.

    Damping is not exposed. The AOUSD deformable schema defines damping alongside the
    moduli, but Isaac Lab does not author it.

To target a specific axial ``E * A`` or bending ``E * I``, invert these
relations to pick the modulus; ``scripts/demos/cables.py`` does this from a
target stiffness and the segment geometry.

.. warning::
    Newton derives **one** stretch/bend stiffness pair for the whole cable, using the
    **mean** segment length as ``L``. Author :attr:`positions` with roughly uniform
    spacing: with a strongly uneven spacing the per-joint stiffness is wrong for the
    outlier segments, since stiffness scales as ``1 / L``. A segment much longer than
    the mean comes out too stiff, and a much shorter one too soft.


Collision
---------

Collision is opt-in through ``collision_props``. When enabled, the importer
applies **adjacent-segment-only** collision filtering: directly connected
segments (which share a joint anchor and would otherwise jitter) are filtered,
while every other pair collides. As a result:

* The cable collides with the ground and with other cables.
* Non-adjacent segments of the **same** cable collide, so a cable can
  self-arrest when it loops back on itself.
* Only immediate neighbours are filtered, matching Newton's cable-pile
  behaviour.

When ``collision_props`` is omitted, the cable imports as a dynamics-only rod
and does not collide.


Runtime State
-------------

:class:`~isaaclab.assets.CableObject` exposes per-segment world state through its
data container and integrates with :class:`~isaaclab.scene.InteractiveScene`,
``scene.get_state`` / ``scene.reset_to``, and the ``reset_scene_to_default``
event term.

* ``cable.data.segment_pose_w`` shape ``(num_instances, num_segments, 7)``,
  position [m] followed by quaternion ``(x, y, z, w)``.
* ``cable.data.segment_velocity_w`` shape ``(num_instances, num_segments, 6)``,
  linear [m/s] followed by angular [rad/s].
* ``cable.data.default_segment_pose_w`` / ``default_segment_velocity_w`` capture
  the spawn state for restoration.

Write per-segment state back with the indexed or masked setters. The masked
form is CUDA-graph capturable:

.. code-block:: python

    cable.write_segment_pose_to_sim_index(
        segment_pose=cable.data.default_segment_pose_w,
    )
    cable.write_segment_velocity_to_sim_index(
        segment_velocity=cable.data.default_segment_velocity_w,
    )

Writes update Newton's maximal-coordinate body state directly (both simulation
states) and flag the affected environments for a solver reset, without running
forward kinematics.


Kit / Fabric Visualization
--------------------------

Cables render in the Kit viewport as ``UsdGeom.BasisCurves``. At render cadence
the curve points are refreshed from Newton's cable segment endpoints so the
rendered shape always matches the simulation. Use the default
``--visualizer kit`` flag, as in the demo.

.. note::
    Curve points are synchronized through **CPU** Fabric because the RTX Hydra
    delegate does not read GPU-backed Fabric arrays for ``BasisCurves``
    (NVBug 6502662). The on-device sync path can be restored once that bug is
    fixed. Periodic curves are skipped by the sync.


Loading Cables from USD
-----------------------

Physics is authored in place on the curve:
:func:`~isaaclab.sim.spawners.shapes.spawn_cable` applies the
``PhysicsCurvesDeformableSimAPI`` schema and binds a deformable-curve material,
and Newton imports the curve natively through ``ModelBuilder.add_usd``. Topology
comes from the curve's own ``points`` and ``curveVertexCounts``; no custom edge
attribute is required, and imported and replicated cables use the same path.

A cable can therefore also be loaded from an external USD (for example one
authored in a DCC tool) via
:class:`~isaaclab.sim.spawners.from_files.UsdFileCfg`, provided the curve in that
USD already carries:

* a single open, linear, nonperiodic ``UsdGeom.BasisCurves`` under the loaded
  prim,
* the ``PhysicsCurvesDeformableSimAPI`` applied schema, and
* a bound deformable-curve material (``PhysicsCurvesDeformableMaterialAPI``)
  supplying ``thickness``, ``density``, ``stretchStiffness``, and
  ``bendStiffness`` in the ``physics:`` namespace.

A raw exported curve without the physics schema and material will not be
recognized as a cable (the importer falls back to a default radius and warns, or
skips the curve). Author it through
:func:`~isaaclab.sim.spawners.shapes.spawn_cable`, or apply the schema and
material to the imported prim before construction.


Limitations
-----------

* **Newton + VBD only.** Other backends and other Newton solvers are not
  supported.
* **One standalone, unwelded cable per object.**
  :class:`~isaaclab.assets.CableObject` requires "one standalone, unwelded cable
  articulation per simulation world": exactly one ``BasisCurves`` prim carrying
  ``PhysicsCurvesDeformableSimAPI`` under ``prim_path``, holding a single open
  curve that is not welded to another cable. Multi-curve ``BasisCurves`` prims,
  periodic (closed) curves, and hard coincident curve-to-curve
  ``PhysicsAttachment`` welds all fail during initialization.
* **Cable ends can be pinned, not clamped.** A ``PhysicsAttachment`` to an xform
  target lowers to a ball joint, so it constrains position only and the cable
  pivots freely at the anchor. Rigid plugs and end fittings that must transfer
  orientation are not representable. The joint is also created only when the
  attachment stiffness is unauthored or infinite; a finite stiffness is kept as
  metadata and no joint is created. Both cases are import warnings rather than
  errors, so the cable initializes normally with the attachment missing. Check
  the importer output when an attachment appears to have no effect.
* **No damping knobs.** The four stiffness moduli are exposed; their damping
  counterparts are not.
* **Uniform point spacing assumed.** One stiffness pair is derived from the mean segment
  length, so uneven spacing mistunes the outlier segments.
* **CPU-only render sync** (NVBug 6502662); periodic curves are not synced.

.. note::
    Topologies the runtime object rejects still simulate: the Newton model is
    built from the whole USD stage, so every curve carrying
    ``PhysicsCurvesDeformableSimAPI`` is imported whether or not a
    :class:`~isaaclab.assets.CableObject` wraps it. Drive them through
    :class:`~isaaclab_newton.physics.NewtonManager` ``get_model()`` /
    ``get_state_0()`` and your own ``newton.selection.ArticulationView``. There
    is no Isaac Lab asset wrapper for those cases.

For the public API, see :class:`~isaaclab.assets.CableObject`,
:class:`~isaaclab.sim.spawners.shapes.CableCfg`, and
:class:`~isaaclab.sim.spawners.materials.CableMaterialCfg`.
