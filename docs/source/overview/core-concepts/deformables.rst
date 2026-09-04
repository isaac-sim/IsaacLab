.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _deformables:

Deformables
===========

A deformable is an asset whose shape changes during simulation. Unlike a rigid body, it has no
rigid root frame: its state is the position and velocity of every node of its simulation mesh.

Isaac Lab models three kinds of deformable, distinguished by the dimensionality of the simulated
geometry:

.. list-table::
    :header-rows: 1
    :widths: 16 28 28 28

    * - Property
      - Volume
      - Surface
      - Cable
    * - Models
      - Soft solids: rubber blocks, teddy bears, organs
      - Cloth, sheets, membranes
      - Ropes, hoses, wires, harnesses
    * - Simulated geometry
      - Tetrahedral mesh (``UsdGeom.TetMesh``)
      - Triangle mesh (``UsdGeom.Mesh``)
      - Open curve (``UsdGeom.BasisCurves``)
    * - Discretization
      - Tetrahedra with Lame elasticity
      - Triangles with stretch, area, and bend stiffness
      - Capsule segments joined by cable joints
    * - Asset class
      - :class:`~isaaclab.assets.DeformableObject`
      - :class:`~isaaclab.assets.DeformableObject`
      - :class:`~isaaclab.assets.CableObject`
    * - Runtime state
      - Nodal positions and velocities
      - Nodal positions and velocities
      - Per-segment poses and velocities
    * - Kinematic targets
      - Yes
      - Backend-dependent, see `Kinematic targets`_
      - No, ends can be pinned by attachment

Volume and surface deformables share one asset class and one authoring pattern; they differ only
in the physics material assigned to them. Cables are a separate asset class with their own
spawner, material, and state layout, and are covered in `Cables`_.

Particle-based materials such as fluids and granular media are not deformables in this sense. They
use a separate asset and solver; see :doc:`physical-backends/newton/using-mpm`.

.. note::
    All three kinds are under active development. On Newton, deformables are implemented in
    :mod:`isaaclab_contrib.deformable` and re-exported through :mod:`isaaclab_newton.assets`.
    Cable support in particular is experimental: its spawner cfg, asset class, and material
    defaults may still change.


Backend support
---------------

.. list-table::
    :header-rows: 1
    :widths: 22 26 26 26

    * - Kind
      - PhysX
      - Newton
      - OvPhysX
    * - Volume
      - Yes
      - Yes, VBD solver only
      - Experimental, CUDA devices only
    * - Surface
      - Yes
      - Yes, VBD solver only
      - Experimental, CUDA devices only
    * - Cable
      - No
      - Yes, VBD solver only
      - No

On Newton, deformables of every kind are simulated by the VBD solver, so a scene containing them
must select a physics cfg whose solver is VBD. For the solver parameters, for running a rigid robot
and a deformable in one scene, and for the tuning workflow, see
:doc:`physical-backends/newton/using-vbd-solver` and :doc:`/source/concepts/coupled_solvers`.

:class:`~isaaclab.assets.DeformableObject` and :class:`~isaaclab.assets.CableObject` are
backend-dispatched factories. Requesting a cable under PhysX or OvPhysX raises a
:exc:`ValueError` at construction, chained from the backend import error, so a misconfigured scene
fails fast instead of loading the curve as inert geometry.

OvPhysX deformables carry further restrictions on node counts and startup cost. See
:doc:`physical-backends/ovphysx/index`.


Volume and surface deformables
------------------------------

Authoring
^^^^^^^^^

A deformable is a mesh spawner with two extra fields:

* ``deformable_props``, a backend-specific ``*DeformableBodyPropertiesCfg``. Setting it is what
  turns the mesh into a deformable at all.
* ``physics_material``, a deformable material cfg. Its type selects volume or surface.

Wrap the spawner in a :class:`~isaaclab.assets.DeformableObjectCfg` to get a runtime asset.
``debug_vis`` draws the kinematic-target markers described in `Kinematic targets`_.

.. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.assets import DeformableObject, DeformableObjectCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

    cfg = DeformableObjectCfg(
        prim_path="/World/env_.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            collision_props=[PhysxCollisionCfg(rest_offset=0.0, contact_offset=0.001)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=PhysxDeformableBodyMaterialCfg(
                youngs_modulus=1.0e5, poissons_ratio=0.4, density=1000.0
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        debug_vis=True,
    )
    cube = DeformableObject(cfg=cfg)

The same object on Newton swaps the two backend cfgs. Collision properties are omitted because
Newton's VBD path does not use a collider on the simulation mesh; contact is resolved from the
particle radius in the material instead.

.. code-block:: python

    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

    youngs_modulus, poissons_ratio = 1.0e5, 0.4

    spawn = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.2, 0.2),
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
        physics_material=NewtonDeformableBodyMaterialCfg(
            k_mu=youngs_modulus / (2.0 * (1.0 + poissons_ratio)),
            k_lambda=youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio)),
            density=1000.0,
        ),
    )

The two backends parameterize elasticity differently: PhysX takes ``youngs_modulus`` and
``poissons_ratio``, Newton takes the Lame parameters ``k_mu`` and ``k_lambda``, converted above.

A surface deformable is authored the same way, with a 2D mesh spawner and a surface material:

.. code-block:: python

    from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

    cloth = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cloth",
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            edge_refinement=8,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=1.0, particle_radius=0.002, tri_ke=5e2, tri_ka=5e2, edge_ke=0.5
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.1)),
    )

The material values above are the tuned ones from ``Isaac-Lift-Cloth-Franka`` rather than the
defaults. For what each parameter does and what it defaults to, see
:doc:`physical-backends/newton/using-vbd-solver`, which also covers cloth self-contact. Self-contact
is off by default, so cloth passes through itself until it is enabled.

``edge_refinement`` sets the simulation resolution for both kinds: the maximum surface edge length
is the bounding-box diagonal divided by this value, and volume deformables reuse it as the
tetrahedralization target. It defaults to ``4.0`` and must be at least ``1.0``. Values near ``1.0``
make tetrahedralization significantly slower.

The mesh spawner rejects combinations that cannot work: ``deformable_props`` together with
``rigid_props`` or ``mass_props``, or with a ``physics_material`` that is not a deformable
material, raise a :exc:`ValueError` at spawn time.

Loading from USD
^^^^^^^^^^^^^^^^

A deformable can also come from a pre-authored asset. Pass ``deformable_props`` and
``physics_material`` to :class:`~isaaclab.sim.spawners.from_files.UsdFileCfg`; the spawner applies
the deformable schema to the loaded prim, or modifies it in place if the prim already carries one.

.. code-block:: python

    cfg_usd = sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        physics_material=PhysxDeformableBodyMaterialCfg(),
        scale=[0.05, 0.05, 0.05],
    )

Volume or surface
^^^^^^^^^^^^^^^^^

There is no explicit type field, and no separate cloth asset class. At authoring time the kind
follows from the material cfg: a material deriving from
:class:`~isaaclab.sim.spawners.materials.SurfaceDeformableBodyMaterialBaseCfg` produces a surface
deformable, and anything else produces a volume deformable. That choice decides what USD is
authored, a ``UsdGeom.TetMesh`` simulation mesh for volume and a triangle ``UsdGeom.Mesh`` copy of
the visual mesh for surface.

At initialization the asset re-derives the kind from the stage. PhysX and OvPhysX read the applied
schema on the bound physics material, falling back to mesh topology when that is inconclusive;
Newton uses topology alone, treating a ``UsdGeom.TetMesh`` under the prim as volume and a plain
``UsdGeom.Mesh`` as surface.

So the material cfg is load-bearing: pairing a volume material with a cloth-shaped mesh authors a
tetrahedralized solid, not a sheet.

Tetrahedralization
^^^^^^^^^^^^^^^^^^

Volume deformables need a tetrahedral simulation mesh. When the spawned prim does not already
contain a ``UsdGeom.TetMesh``, Isaac Lab generates one, which requires the optional
``tetrahedralization`` dependencies on every backend:

.. code-block:: bash

    uv sync --inexact --extra tetrahedralization

    # With the legacy installer.
    ./isaaclab.sh -i tetrahedralization

Surface deformables never need it, and neither do volume deformables loaded from a USD that already
ships a pre-tetrahedralized ``UsdGeom.TetMesh`` under the deformable prim.

Runtime state
^^^^^^^^^^^^^

The state of a deformable is nodal, expressed in the simulation world frame. ``N`` below is
:attr:`~isaaclab.assets.DeformableObject.max_sim_vertices_per_body`.

* ``data.nodal_pos_w`` and ``data.nodal_vel_w``, shape ``(num_instances, N)`` of ``vec3f``,
  positions [m] and velocities [m/s].
* ``data.nodal_state_w``, the same data as ``(num_instances, N)`` of ``vec6f``.
* ``data.default_nodal_state_w``, the spawn state, used for resets.
* ``data.root_pos_w`` and ``data.root_vel_w``, shape ``(num_instances,)``. These are *derived*
  quantities, computed as the mean over the simulation nodes. There is no root orientation.

Write state back with the indexed setters. :meth:`~isaaclab.assets.DeformableObject.transform_nodal_pos`
applies a pose offset relative to the current nodal mean, which is how a spawn-state buffer is
scattered across environment origins.

.. code-block:: python

    nodal_state = cube.data.default_nodal_state_w.torch[env_ids].clone()
    nodal_state[..., :3] = cube.transform_nodal_pos(nodal_state[..., :3], pos, quat)
    cube.write_nodal_state_to_sim_index(nodal_state, env_ids=env_ids)
    cube.reset(env_ids=env_ids)

Like every other asset, a :class:`~isaaclab.assets.DeformableObject` integrates with
:class:`~isaaclab.scene.InteractiveScene`, ``scene.get_state`` / ``scene.reset_to``, and the
``reset_scene_to_default`` event term.

.. note::
    The unsuffixed writers (``write_nodal_state_to_sim`` and friends) are deprecated in favor of the
    ``_to_sim_index`` and ``_to_sim_mask`` variants. The mask variants are CUDA-graph capturable.

Kinematic targets
^^^^^^^^^^^^^^^^^

Individual nodes can be driven kinematically instead of being solved, which is how a deformable is
pinned or dragged. Targets are written as ``(num_instances, N, 4)``: the target position [m]
followed by a flag that is ``0.0`` for a kinematically driven node and ``1.0`` for a free node.

.. code-block:: python

    targets = cube.data.nodal_kinematic_target.torch.clone()
    targets[:, 0, :3] = anchor_pos    # drive node 0
    targets[:, 0, 3] = 0.0            # mark it kinematic
    cube.write_nodal_kinematic_target_to_sim_index(targets)
    cube.write_data_to_sim()

:meth:`~isaaclab.assets.DeformableObject.write_data_to_sim` is a no-op on PhysX and OvPhysX, which
apply targets on write. On Newton it is what flushes the target buffer into the particle state, so
call it every step while nodes are pinned.

Backend coverage differs. On PhysX and OvPhysX kinematic targets are volume-only, and calling the
setter on a surface deformable raises a :exc:`ValueError`. Newton accepts them on surface
deformables as well. Code that must run on more than one backend should not rely on pinning cloth.


Cables
------

A cable is a 1D rod: a single open ``UsdGeom.BasisCurves`` prim carrying the
``PhysicsCurvesDeformableSimAPI`` schema, simulated as a chain of per-segment capsule bodies joined
by cable joints.

Cable authoring
^^^^^^^^^^^^^^^

A cable is configured with a :class:`~isaaclab.sim.spawners.shapes.CableCfg` plus a
:class:`~isaaclab.sim.spawners.materials.CableMaterialCfg`. Adjacent pairs in ``positions`` become
segments, each materialized as a capsule of diameter ``thickness`` and joined to its neighbor by a
cable joint. ``N`` control points produce ``N-1`` segment bodies and ``N-2`` joints; the root
segment is free-floating.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``positions``
      - Control points in the cable-local frame [m]. Requires at least **three** finite points,
        with consecutive points separated by more than ``1e-8`` m.
    * - ``physics_material``
      - Required :class:`~isaaclab.sim.spawners.materials.CableMaterialCfg`; see
        `Cable material parameters`_. Thickness is also written to the curve's ``widths``
        attribute so the visual radius matches the physics.
    * - ``collision_props``
      - Optional collision properties. When omitted, the cable is collision-free; when set
        (typically ``[UsdPhysicsCollisionCfg(collision_enabled=True)]``), the cable collides with
        the ground and other cables. See `Cable collision`_.
    * - ``visual_material``
      - Optional :class:`~isaaclab.sim.spawners.materials.VisualMaterialCfg` for the curve's
        appearance.
    * - ``visual_material_path``
      - Sub-path of the visual material under the cable geometry prim. Defaults to ``"material"``.
    * - ``physics_material_path``
      - Sub-path of the physics material under the cable geometry prim. Defaults to
        ``"physics_material"``.

.. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.assets import CableObject, CableObjectCfg

    cable = CableObject(
        cfg=CableObjectCfg(
            prim_path="/World/Env_0/Cable",
            spawn=sim_utils.CableCfg(
                positions=[(index * 0.1, 0.0, 0.0) for index in range(10)],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2)),
                physics_material=sim_utils.CableMaterialCfg(
                    thickness=0.03,
                    density=1000.0,
                    stretch_stiffness=1.0e9,
                    bend_stiffness=1.0e6,
                ),
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
    )

This authors a straight 0.9 m red cable along the cable-local x-axis: 9 capsule segments (0.1 m
long, 0.03 m diameter) joined by 8 cable joints. The inherited ``init_state`` sets the cable root's
spawn pose in each environment.

Cable material parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~isaaclab.sim.spawners.materials.CableMaterialCfg` defines the cable's geometry and
stiffness. All values are validated at author time; nonfinite or out-of-range values raise a
:exc:`ValueError` before any prim is created.

The stiffness fields are elastic moduli, not joint stiffnesses. Each rod joint gets its own
discretized stiffness, using that joint's dual rest length ``L = 0.5 * (L_parent + L_child)``,
where ``L_parent`` and ``L_child`` are the rest lengths of the two segments it spans.

.. list-table::
    :header-rows: 1
    :widths: 24 76

    * - Parameter
      - Description
    * - ``thickness``
      - Full cable thickness (diameter) [m]. Must be finite and positive. Used as
        ``radius = thickness / 2`` for the capsule cross-section, the bending second moment of
        area, and the collision radius. Default ``0.001``.
    * - ``density``
      - Cable density [kg/m^3]. Must be finite and positive. Per-segment mass is derived from the
        density and the capsule volume. Default ``1000.0``.
    * - ``stretch_stiffness``
      - Axial (stretch) elastic modulus ``E`` [Pa]. Must be finite and nonnegative. Becomes the
        per-joint axial stiffness ``E * A / L``, where ``A`` is the circular cross-section area.
        Higher values reduce elongation. Default ``1.0e9``.
    * - ``bend_stiffness``
      - Bending elastic modulus ``E`` [Pa]. Must be finite and nonnegative. Becomes the per-joint
        bend stiffness ``E * I / L``, where ``I`` is the second moment of area of the circular
        cross-section. ``0.0`` gives a limp rope; increase for a stiff hose or wire. Default
        ``1.0e6``.
    * - ``shear_stiffness``
      - Transverse shear elastic modulus [Pa]. Must be finite and nonnegative. Optional: when left
        at ``None`` the attribute is not authored and the solver reuses ``stretch_stiffness``.
        Default ``None``.
    * - ``twist_stiffness``
      - Torsional elastic modulus [Pa]. Must be finite and nonnegative. Optional: when left at
        ``None`` the attribute is not authored and the solver reuses the bend *structural*
        stiffness. Default ``None``.

.. note::
    A cable joint has four degrees of freedom: linear stretch and shear, and angular bend and
    twist. Leaving ``shear_stiffness`` or ``twist_stiffness`` unset does **not** mean the cable has
    no shear or twist resistance; it means the solver falls back to the stretch and bend values.
    Authoring ``0.0`` is distinct from leaving them unset: it removes that resistance.

    The two fallbacks are not symmetric. Shear reuses the stretch modulus, applied to the same
    cross-section area. Twist reuses the bend *structural* stiffness ``E * I / L``, whereas an
    explicitly authored ``twist_stiffness`` is applied to the polar moment ``J = 2 * I``. Setting
    ``twist_stiffness = bend_stiffness`` therefore gives twice the fallback stiffness, not the same
    value.

    Damping is not exposed. The rod joints have damping, but the USD curve-material schema Isaac
    Lab authors has no attribute for it.

To target a specific axial ``E * A`` or bending ``E * I``, invert these relations to pick the
modulus; ``scripts/demos/cables.py`` does this from a target stiffness and the segment geometry.

Cable collision
^^^^^^^^^^^^^^^

Collision is opt-in through ``collision_props``. When enabled, the importer applies
**adjacent-segment-only** filtering: directly connected segments, which share a joint anchor and
would otherwise jitter, are filtered, while every other pair collides. As a result:

* The cable collides with the ground and with other cables.
* Non-adjacent segments of the **same** cable collide, so a cable can self-arrest when it loops
  back on itself.

When ``collision_props`` is omitted, the cable imports as a dynamics-only rod and does not collide.

Cable runtime state
^^^^^^^^^^^^^^^^^^^

:class:`~isaaclab.assets.CableObject` exposes per-segment world state and integrates with
:class:`~isaaclab.scene.InteractiveScene`, ``scene.get_state`` / ``scene.reset_to``, and the
``reset_scene_to_default`` event term.

* ``cable.data.segment_pose_w``, shape ``(num_instances, num_segments)`` of ``wp.transformf``,
  position [m] followed by quaternion ``(x, y, z, w)``. The Torch view has a trailing dimension
  of 7.
* ``cable.data.segment_velocity_w``, shape ``(num_instances, num_segments)`` of
  ``wp.spatial_vectorf``, linear [m/s] followed by angular [rad/s]. The Torch view has a trailing
  dimension of 6.
* ``cable.data.default_segment_pose_w`` and ``default_segment_velocity_w``, the spawn state.

Write per-segment state back with the indexed or masked setters; the masked form is CUDA-graph
capturable.

.. code-block:: python

    cable.write_segment_pose_to_sim_index(segment_pose=cable.data.default_segment_pose_w)
    cable.write_segment_velocity_to_sim_index(segment_velocity=cable.data.default_segment_velocity_w)

Writes update the maximal-coordinate body state directly, for both simulation states, and flag the
affected environments for a solver reset.

Loading cables from USD
^^^^^^^^^^^^^^^^^^^^^^^

Physics is authored in place on the curve: :func:`~isaaclab.sim.spawners.shapes.spawn_cable` applies
the ``PhysicsCurvesDeformableSimAPI`` schema and binds a deformable-curve material, and the curve is
imported natively. Topology comes from the curve's own ``points`` and ``curveVertexCounts``; no
custom edge attribute is required.

A cable can therefore also be loaded from an external USD, for example one authored in a DCC tool,
via :class:`~isaaclab.sim.spawners.from_files.UsdFileCfg`, provided the curve already carries:

* a single open, linear, nonperiodic ``UsdGeom.BasisCurves`` under the loaded prim,
* the ``PhysicsCurvesDeformableSimAPI`` applied schema, and
* a bound deformable-curve material (``PhysicsCurvesDeformableMaterialAPI``) supplying
  ``thickness``, ``density``, ``stretchStiffness``, and ``bendStiffness`` in the ``physics:``
  namespace.

Two failure modes follow from that list. A curve without the sim schema is not part of the
deformable import at all and is silently skipped. A curve that has the schema but no resolvable
thickness is imported with a default radius and a warning. Author cables through
:func:`~isaaclab.sim.spawners.shapes.spawn_cable`, or apply the schema and material to the imported
prim before construction.

Cable rendering
^^^^^^^^^^^^^^^

Cables render in the Kit viewport as ``UsdGeom.BasisCurves``. At render cadence the curve points are
refreshed from the cable segment endpoints so the rendered shape always matches the simulation.

.. note::
    Curve points are synchronized through **CPU** Fabric because the RTX Hydra delegate does not
    read GPU-backed Fabric arrays for ``BasisCurves`` (NVBug 6502662). Periodic curves are skipped
    by the sync.

Cable limitations
^^^^^^^^^^^^^^^^^

* **One standalone, unwelded cable per object.** :class:`~isaaclab.assets.CableObject` requires
  exactly one ``BasisCurves`` prim carrying ``PhysicsCurvesDeformableSimAPI`` under ``prim_path``,
  holding a single open curve that is not welded to another cable. Multi-curve ``BasisCurves``
  prims, periodic (closed) curves, and hard coincident curve-to-curve ``PhysicsAttachment`` welds
  all fail during initialization.
* **Cable ends can be pinned, not clamped.** A ``PhysicsAttachment`` to an xform target lowers to a
  ball joint, so it constrains position only and the cable pivots freely at the anchor. Rigid plugs
  and end fittings that must transfer orientation are not representable.
* **No damping knobs.** The four stiffness moduli are exposed; their damping counterparts are not.
* **CPU-only render sync** (NVBug 6502662); periodic curves are not synced.
* **Culled by Isaac RTX scene partitioning** once the cable deforms beyond its initial extent
  (OMPE-105749). See :ref:`known-issues-animated-curve-scene-partition`.

.. note::
    An attachment joint is created only when the attachment stiffness is unauthored or infinite. A
    finite stiffness is kept as metadata and no joint is created. Both cases are import warnings
    rather than errors, so the cable initializes normally with the attachment missing. Check the
    importer output when an attachment appears to have no effect.

.. note::
    Topologies the runtime object rejects still simulate: the physics model is built from the whole
    USD stage, so every curve carrying ``PhysicsCurvesDeformableSimAPI`` is imported whether or not
    a :class:`~isaaclab.assets.CableObject` wraps it. Drive them through
    :class:`~isaaclab_newton.physics.NewtonManager` ``get_model()`` / ``get_state_0()`` and your own
    ``newton.selection.ArticulationView``. There is no Isaac Lab asset wrapper for those cases.


Material parameters in practice
-------------------------------

Each animation below runs the same scene three times, changing one material parameter and holding
everything else fixed. The labels give the value used in each run.

Volume
^^^^^^

.. figure:: ../../_static/deformables/volume_youngs_modulus.gif
    :align: center
    :alt: A soft ball dropped on a table at three values of Young's modulus

    **Young's modulus**: overall stiffness. At ``1e3`` Pa the ball collapses into a pancake on
    impact; at ``1e5`` Pa it barely deforms. This is the first knob to reach for when a soft body
    is too floppy or too rigid.

.. figure:: ../../_static/deformables/volume_poissons_ratio.gif
    :align: center
    :alt: A soft ball dropped on a table at three values of Poisson's ratio

    **Poisson's ratio**: how strongly the material preserves volume under compression. At ``0.10``
    the ball squashes without spreading. At ``0.49`` it is nearly incompressible, so the same
    squash has to go somewhere and pushes outward into a wide bulge.

.. figure:: ../../_static/deformables/volume_particle_radius.gif
    :align: center
    :alt: A soft ball resting on a table at three values of particle radius

    **Particle radius**: the contact thickness around each simulation node, not a material
    property. A larger radius detects contact further from the surface, so the ball rests visibly
    higher off the table and cannot be compressed as thin. Too small and contacts are missed or
    detected late; too large relative to the mesh resolution and they start too early.

Surface
^^^^^^^

.. figure:: ../../_static/deformables/cloth_stretch_stiffness.gif
    :align: center
    :alt: A cloth sheet draped over a roller at three values of stretch stiffness

    **Stretch stiffness**: resistance to in-plane elongation. The low value lets the sheet stretch
    and sag under its own weight; the high value holds it near its rest length.

.. figure:: ../../_static/deformables/cloth_bend_stiffness.gif
    :align: center
    :alt: A cloth sheet falling from a roller at three values of bend stiffness

    **Bend stiffness**: resistance to folding. At ``1e-2`` the sheet crumples into a loose heap; at
    ``1e0`` it keeps large, stiff folds and stays draped over the roller. This is the difference
    between silk and canvas.

Cable
^^^^^

.. figure:: ../../_static/deformables/cable_bend_stiffness.gif
    :align: center
    :alt: A cable spanning two posts at three values of bend stiffness

    **Bend stiffness**: at ``1e0`` the cable droops between its supports like slack rope; at
    ``1e2`` it holds itself straight like a stiff hose.

.. figure:: ../../_static/deformables/cable_twist_stiffness.gif
    :align: center
    :alt: A weighted hanging cable twisted at the top, at three values of twist stiffness

    **Twist stiffness**: a weighted cable is twisted at its anchor. At ``1e-1`` the twist is
    absorbed locally and the cable hangs straight. At ``1e1`` torsion is carried along the rod
    until it buckles and coils into a helix. Set this explicitly when a cable should resist
    winding, since leaving it unset falls back to the bend value as described above.


Demos and tasks
---------------

Run a demo first to confirm that the spawner, solver, and visualizer all work in your environment.

.. list-table::
    :header-rows: 1
    :widths: 20 44 36

    * - Kind
      - Demo
      - Tasks
    * - Volume
      - ``scripts/demos/deformables.py``
      - ``Isaac-Lift-Soft-Franka``, ``Isaac-Lift-Soft-Franka-Camera``
    * - Surface
      - ``scripts/demos/deformables.py``
      - ``Isaac-Lift-Cloth-Franka``, ``Isaac-Lift-Cloth-Franka-Camera``
    * - Cable
      - ``scripts/demos/cables.py``
      - ``Isaac-Lift-Cable-Franka``, ``Isaac-Lift-Cable-Franka-Camera``

.. code-block:: bash

    # Volume and surface deformables falling onto a ground plane.
    uv run --extra isaacsim --extra tetrahedralization python scripts/demos/deformables.py

    # A pile of cables that collide and settle. Newton VBD only.
    uv run --extra isaacsim python scripts/demos/cables.py

    # A larger cable pile, without a visualizer, stopping after a fixed number of steps.
    uv run python scripts/demos/cables.py --visualizer none --num_cables 40 --num_segments 15 --max_steps 500

``scripts/environments/state_machine/lift_franka_soft.py`` drives ``Isaac-Lift-Soft-Franka`` with a
scripted state machine, which is a useful starting point for a deformable manipulation task.


Related
-------

* :ref:`tutorial-interact-deformable-object` walks through a volume deformable step by step.
* :doc:`physical-backends/newton/using-vbd-solver` covers the VBD solver parameters, the Newton
  material tables, cloth self-contact, and the tuning workflow.
* :doc:`/source/concepts/coupled_solvers` covers running a rigid robot and a deformable in one
  scene.
* :ref:`migrating-deformables` covers the Isaac Lab 3.0 deformable API changes.
* :doc:`physical-backends/ovphysx/index` covers the OvPhysX deformable limitations.
* :doc:`/source/api/lab/isaaclab.assets` and :doc:`/source/api/lab/isaaclab.sim.spawners` are the
  API references for the asset, spawner, and material classes used here.
