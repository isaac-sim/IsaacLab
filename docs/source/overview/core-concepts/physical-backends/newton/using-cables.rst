.. _newton-using-cables:

Using Cables
============

Isaac Lab exposes 1D rod / cable assets through Newton's
:meth:`newton.ModelBuilder.add_rod_graph`. A cable is spawned as a
``UsdGeomBasisCurves`` prim, and the cable's physics (per-segment capsules,
inter-segment cable joints, stretch / bend stiffness, damping, density) is
materialized at Newton model-build time by a contrib replicate hook.

Cable support is experimental. The spawner cfg, contrib asset class, registry
entry, and material defaults may change while Newton cable support is under
active development.

.. note::
    Cables are currently **only supported on the Newton physics backend**.
    The spawner authors valid USD on any backend (so the scene loads in PhysX
    or PhysX-Fabric viewports), but the resulting cable is not registered with
    a PhysX articulation. :class:`~isaaclab.sim.spawners.shapes.CableCfg`
    requires ``physics_material`` to be a
    :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg`
    and rejects ``rigid_props`` / ``mass_props`` up front.


Quick Start: The Cable Demo
---------------------------

Before adding cables to a task, run the standalone demo to confirm that the
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
plus a Newton-specific physics material. The cfg's ``positions`` field is a
list of at least two control points in the cable's local frame; adjacent pairs
become individual rod segments, each materialized as a capsule body of
diameter ``width`` and joined to its neighbour by a Newton cable joint.

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
      - Default: ``0.0`` [N·m^2]. Bending and twisting stiffness EI. ``0.0``
        produces a fully limp rope; increase for stiffer hoses or wires.
    * - ``stretch_damping``
      - Default: ``0.0`` [N·s/m]. Per-joint axial damping. Increase to remove
        post-contact stretch oscillations.
    * - ``bend_damping``
      - Default: ``0.0`` [N·m·s/rad]. Per-joint bend / twist damping.
    * - ``density``
      - Default: ``1500.0`` [kg/m^3]. Material density. The cable replicate
        hook converts this to per-segment mass via the capsule volume
        ``pi * radius^2 * segment_length * density`` and passes it through
        :class:`newton.ModelBuilder.ShapeConfig` to
        :meth:`newton.ModelBuilder.add_rod_graph`.


Spawner Parameters
------------------

:class:`~isaaclab.sim.spawners.shapes.CableCfg` fields specific to cables
(beyond the inherited :class:`~isaaclab.sim.spawners.shapes.ShapeCfg` slots):

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Parameter
      - Description
    * - ``positions``
      - List of control points in cable-local frame [m]. Must contain at least
        two points. Adjacent pairs define one cable segment each, so a list of
        ``N`` points produces ``N-1`` rod segments and ``N-2`` cable joints
        plus one root joint anchoring the rod.
    * - ``width``
      - Capsule diameter for every segment [m]. The same value is also written
        to the ``UsdGeomBasisCurves`` ``widths`` attribute so the visual
        thickness matches the physics.
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
attributes. ``collision_props`` is required so that
:class:`UsdPhysics.CollisionAPI` can author a valid binding for the physics
material.


Kit / Fabric Visualization
--------------------------

The cable replicate hook places one ``UsdGeomBasisCurves`` prim per cable per
environment. The Newton VBD manager keeps these curves in sync with the
simulated body transforms by reconstructing the control points from
``newton.State.body_q`` every render frame. This sync runs on the **CPU Fabric
device** because Kit / Hydra reads curve points from the CPU Fabric bucket for
runtime-spawned ``UsdGeomBasisCurves``. If your visualizer skips curves at
runtime, prefer the default ``--visualizer kit`` flag used by the demo.

A ``reset()`` call on a :class:`~isaaclab_contrib.cable.CableObject` does
**not** snap control points back to their initial positions: cables have no
nodal snap-back, only internal-buffer reset. To re-pose cables, write directly
into ``body_q`` or recreate the scene.


Limitations
-----------

* Newton-only. PhysX has no cable joint, so the cable will load as inert
  geometry under a PhysX backend.
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
