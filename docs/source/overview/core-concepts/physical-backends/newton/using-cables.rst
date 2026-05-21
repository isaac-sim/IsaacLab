.. _newton-using-cables:

Using Cables
============

Isaac Lab exposes 1D rod / cable assets through Newton's
:meth:`newton.ModelBuilder.add_rod_graph`. A cable is spawned as a continuous USD curve prim, and is simulated in Newton as a passive articulated rigid body chain with per-segment capsules,
inter-segment cable joints, stretch / bend stiffness, damping, and density.

Cable support is experimental. The spawner cfg, contrib asset class, registry
entry, and material defaults may change while Newton cable support is under
active development.

.. note::
    Cables are **only supported on the Newton physics backend**.
    :func:`~isaaclab.sim.spawners.shapes.spawn_cable` raises :class:`RuntimeError`
    when invoked under any other backend (e.g. PhysX), so a misconfigured scene
    fails fast instead of loading the curve as inert geometry.
    :class:`~isaaclab.sim.spawners.shapes.CableCfg` also requires
    ``physics_material`` to be a
    :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg`
    and rejects ``rigid_props`` / ``mass_props`` up front.


Quick Start: The Cable Demo
---------------------------

Before adding cables to a task, it is a good sanity check to run the standalone demo to confirm that the
spawner, the cable replicate hook, the VBD solver, and the Kit / Fabric
viewport sync are all working in your environment:

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/cables.py
    ./isaaclab.sh -p scripts/demos/cables.py --num_cables 40

The demo spawns a pile of randomly oriented cables onto a ground plane under
the Newton VBD solver. Source: ``scripts/demos/cables.py``.


Authoring a Cable
-----------------

A cable is configured as a :class:`~isaaclab.sim.spawners.shapes.CableCfg`
plus a Newton-specific physics material. Adjacent pairs in ``positions`` become
individual rod segments, each materialized as a capsule body of diameter
``width`` and joined to its neighbour by a Newton cable joint.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``positions``
      - Control points in cable-local frame [m]. Must contain at least two
        points; ``N`` points produce ``N-1`` rod segments and ``N-2`` cable
        joints plus one root joint anchoring the rod.
    * - ``width``
      - Capsule diameter for every segment [m]. Also written to the
        ``UsdGeomBasisCurves`` ``widths`` attribute so the visual thickness
        matches the physics.
    * - ``physics_material``
      - Required :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg`;
        see `Cable Material Parameters`_ below.
    * - ``collision_props``
      - Required :class:`~isaaclab.sim.schemas.CollisionPropertiesCfg`. Applies
        :class:`UsdPhysics.CollisionAPI` to the curve prim so the physics
        material binding is valid. (Has no PhysX runtime effect since cables
        are Newton-only.)
    * - ``visual_material``
      - Optional :class:`~isaaclab.sim.spawners.materials.VisualMaterialCfg` for
        the curve's appearance.
    * - ``visual_material_path``
      - Default: ``"visual_material"``. Sub-path under ``{prim_path}/geometry``.
        Overrides :attr:`ShapeCfg.visual_material_path` so visual and physics
        materials don't collide at the same sub-path.
    * - ``physics_material_path``
      - Default: ``"physics_material"``. Same as above for the Newton physics
        material.

``rigid_props`` and ``mass_props`` are inherited from
:class:`~isaaclab.sim.spawners.shapes.ShapeCfg` but must remain ``None``:
:func:`~isaaclab.sim.spawners.shapes.spawn_cable` raises ``ValueError`` if
either is set, because cable mass and rigid-body properties come from the
material density and the rod-graph topology — not from per-prim USD physics
attributes.

.. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

    cable_spawn = sim_utils.CableCfg(
        positions=[(i * 0.1, 0.0, 0.0) for i in range(10)],
        width=0.03,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2)),
        physics_material=NewtonCableMaterialCfg(
            stretch_stiffness=1.0e6,
            bend_stiffness=1.0e-4,
            stretch_damping=1.0e-4,
            bend_damping=1.0e-4,
            density=1000.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

Wrap the spawner in a :class:`~isaaclab_contrib.cable.CableObjectCfg` to get a
runtime asset that can be reset and inspected through
:class:`~isaaclab_newton.assets.articulation.Articulation` state:

.. code-block:: python

    from isaaclab_contrib.cable import CableObject, CableObjectCfg

    cable = CableObject(
        cfg=CableObjectCfg(
            prim_path="/World/Origin/Cable",
            spawn=cable_spawn,
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
    )

The :class:`~isaaclab_contrib.cable.CableObject` constructor appends a
:class:`~isaaclab_contrib.cable.CableRegistryEntry` to the contrib cable
registry. The Newton VBD manager installs a per-world builder hook that walks
this registry on each replicate and calls
:meth:`newton.ModelBuilder.add_rod_graph` so the cable is materialized once per
environment. See :doc:`newton-manager-abstraction` for the registry / hook
pattern that the deformable contrib package also follows.


Picking a Solver
----------------

Cables are integrated as Newton articulations, but they currently must be
simulated under a solver that knows how to step
:attr:`newton.JointType.CABLE` joints. The VBD manager in
:mod:`isaaclab_contrib.deformable` ships with that support:

.. code-block:: python

    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

    physics_cfg = NewtonCfg(
        solver_cfg=VBDSolverCfg(iterations=20),
        num_substeps=8,
    )
    physics_cfg.model_cfg = NewtonModelCfg(
        shape_material_ke=1.0e3,
        shape_material_kd=1.0e1,
        shape_material_mu=1.0,
    )

A cable-only scene can use a bare
:class:`~isaaclab_contrib.deformable.VBDSolverCfg`. Mixed rigid + cable scenes
(robot manipulating a cable) should use a coupled solver — see
:doc:`using-vbd-solver`.


Cable Material Parameters
-------------------------

:class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg`
exposes the rod material. Stiffness values are EA / EI quantities and are
normalized internally by Newton by the segment length.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``stretch_stiffness``
      - Default: ``1.0e9`` [N]. Axial stiffness EA. Higher values reduce
        cable elongation but require more solver iterations or substeps.
    * - ``bend_stiffness``
      - Default: ``0.0`` [N·m²]. Bending and twisting stiffness EI. ``0.0``
        produces a fully limp rope; increase for stiffer hoses or wires.
    * - ``stretch_damping``
      - Default: ``0.0`` [N·s/m]. Per-joint axial damping. Increase to remove
        post-contact stretch oscillations.
    * - ``bend_damping``
      - Default: ``0.0`` [N·m·s/rad]. Per-joint bend / twist damping.
    * - ``density``
      - Default: ``1500.0`` [kg/m³]. Material density. The cable replicate
        hook converts this to per-segment mass via the capsule volume
        ``pi * radius² * segment_length * density`` and passes it through
        :class:`newton.ModelBuilder.ShapeConfig` to
        :meth:`newton.ModelBuilder.add_rod_graph`.


Kit / Fabric Visualization
--------------------------

The cable replicate hook places one ``UsdGeomBasisCurves`` prim per cable per
environment. The Newton VBD manager keeps these curves in sync with the
simulated body transforms by reconstructing the control points from
``newton.State.body_q`` every render frame. This sync runs on the **CPU Fabric
device** because Kit / Hydra reads curve points from the CPU Fabric bucket for
runtime-spawned ``UsdGeomBasisCurves``. If your visualizer skips curves at
runtime, prefer the default ``--visualizer kit`` flag used by the demo.

A ``reset()`` call on a :class:`~isaaclab_contrib.cable.CableObject` snaps
each environment's cable bodies back to the spawn pose stored in
``newton.Model.body_q`` and zeroes both ``state.body_qd`` and the AVBD
``solver.body_inertia_q`` buffer. The implicit-velocity buffer
``solver.body_q_prev`` is also restored to the rest pose — without this,
AVBD's ``(body_q - body_q_prev) / dt`` velocity estimate would emit ~700 m/s
spurious velocities the step after a snap-back. Joint state and AVBD
penalty / Dahl buffers are intentionally left alone: they are either global
to the world or would require joint offsets in the registry to slice
per-env, and the body-side reset is sufficient to keep post-reset dynamics
bounded in practice.


Loading Cables from USD
-----------------------

In addition to the procedural :class:`~isaaclab.sim.spawners.shapes.CableCfg`
path, a cable can be loaded from an arbitrary USD via
:class:`~isaaclab.sim.spawners.from_files.UsdFileCfg`. The USD must contain
exactly one ``UsdGeomBasisCurves`` prim anywhere under the loaded template
prim — :class:`~isaaclab_contrib.cable.CableObject` walks the template
prim's descendants with ``Usd.PrimRange`` and raises
``NotImplementedError`` if more than one curve is found (multi-curve cables
under a single :class:`CableObject` are not supported yet).

.. code-block:: python

    from isaaclab_contrib.cable import CableObject, CableObjectCfg
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

    import isaaclab.sim as sim_utils

    cable = CableObject(
        cfg=CableObjectCfg(
            prim_path="/World/Origin/Cable",
            spawn=sim_utils.UsdFileCfg(
                usd_path="path/to/cable.usda",
                physics_material=NewtonCableMaterialCfg(density=100.0),
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
    )

The curve prim must author three attributes:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Attribute
      - Description
    * - ``point3f[] points``
      - Control points in the curve prim's local frame [m]. The curve prim's
        xform is baked into these positions at registration time, so the
        replicate hook only needs to apply the per-env transform.
    * - ``float[] widths``
      - One width per control point [m]. For now, only the first entry is
         read — it defines the capsule diameter for every segment.
    * - ``int2[] connections``
      - Edge topology — each ``Vec2i`` lists the indices of one segment's two
        endpoint control points. :func:`~isaaclab.sim.spawners.shapes.spawn_cable`
        writes a linear chain ``[(0,1), (1,2), ...]`` automatically;
        user-imported curve USDs must author this attribute explicitly, since
        ``connections`` is not part of the ``UsdGeomBasisCurves`` schema and
        cannot be inferred from the curve's vertex counts.

The Newton cable material is taken from the spawner's ``physics_material``
binding on the curve prim. If no Newton cable material is bound, the
:class:`~isaaclab_contrib.cable.CableRegistryEntry` defaults are used.


Limitations
-----------

* Newton-only. PhysX has no cable joint, so
  :func:`~isaaclab.sim.spawners.shapes.spawn_cable` raises :class:`RuntimeError`
  under a non-Newton backend rather than authoring inert geometry.
* No actuators. :class:`~isaaclab_contrib.cable.CableObjectCfg` overrides
  ``actuators`` to ``{}``; per-cable stiffness is treated as material, not as
  a controllable joint. The inherited
  ``logger.warning("Not all actuators are configured!")`` is expected and
  harmless.
* :meth:`newton.eval_fk` has no
  :attr:`newton.JointType.CABLE` case at present. The VBD manager
  works around this by building a non-cable articulation mask in
  :meth:`~isaaclab_contrib.deformable.vbd_manager.NewtonVBDManager._build_non_cable_articulation_mask`
  and overriding
  :meth:`~isaaclab_contrib.deformable.vbd_manager.NewtonVBDManager.forward`
  so Kit-triggered pre-render FK passes don't collapse rod segments onto their
  parent anchors. Once Newton patches cable joints in ``eval_fk``, that mask
  and override can be removed.
* Self-contact between cable segments uses the rigid contact pipeline
  (``shape_material_ke`` / ``kd`` / ``mu`` on
  :class:`~isaaclab_contrib.deformable.NewtonModelCfg`), not VBD particle
  self-contact. For dense cable piles, lower ``shape_material_ke``, raise
  ``shape_material_kd``, and increase
  :attr:`~isaaclab_contrib.deformable.VBDSolverCfg.rigid_body_contact_buffer_size`
  before raising iterations.

For implementation details of the cable registry, replicate hook, and Fabric
curve sync, see :class:`~isaaclab_contrib.cable.CableObject` and the
deformable contrib :doc:`newton-manager-abstraction` guide.
