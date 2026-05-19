# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for cable-endpoint ↔ rigid-body attachments."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first (required for the E2E test below)."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from typing import Literal

import pytest


def test_cable_attachment_cfg_defaults_and_types():
    """CableAttachmentCfg accepts head/tail anchors and exposes the documented defaults."""
    from isaaclab_contrib.cable import CableAttachmentCfg

    cfg = CableAttachmentCfg(target_prim_path="/World/Plug001", cable_anchor="tail")
    assert cfg.target_prim_path == "/World/Plug001"
    assert cfg.cable_anchor == "tail"
    assert cfg.cable_local_pos == (0.0, 0.0, 0.0)
    assert cfg.cable_local_quat == (0.0, 0.0, 0.0, 1.0)
    assert cfg.target_local_pos == (0.0, 0.0, 0.0)
    assert cfg.target_local_quat == (0.0, 0.0, 0.0, 1.0)

    cfg2 = CableAttachmentCfg(
        target_prim_path="/Foo",
        cable_anchor="head",
        cable_local_pos=(1.0, 2.0, 3.0),
        cable_local_quat=(0.5, 0.5, 0.5, 0.5),
        target_local_pos=(4.0, 5.0, 6.0),
        target_local_quat=(0.7071, 0.7071, 0.0, 0.0),
    )
    assert cfg2.cable_anchor == "head"
    assert cfg2.cable_local_pos == (1.0, 2.0, 3.0)
    assert cfg2.cable_local_quat == (0.5, 0.5, 0.5, 0.5)
    assert cfg2.target_local_pos == (4.0, 5.0, 6.0)
    assert cfg2.target_local_quat == (0.7071, 0.7071, 0.0, 0.0)


def test_cable_object_cfg_attachments_field_default_empty():
    """CableObjectCfg exposes an `attachments` list field that defaults to empty."""
    from isaaclab_contrib.cable import CableAttachmentCfg
    from isaaclab_contrib.cable.cable_object_cfg import CableObjectCfg

    cfg = CableObjectCfg(prim_path="/World/Cable001")
    assert hasattr(cfg, "attachments"), "CableObjectCfg must expose an `attachments` field"
    assert cfg.attachments == []

    cfg2 = CableObjectCfg(
        prim_path="/World/Cable001",
        attachments=[CableAttachmentCfg(target_prim_path="/World/Plug001", cable_anchor="tail")],
    )
    assert len(cfg2.attachments) == 1
    assert cfg2.attachments[0].target_prim_path == "/World/Plug001"
    assert cfg2.attachments[0].cable_anchor == "tail"


def test_cable_registry_records_head_tail_body_indices():
    """After add_cable_entry_to_builder runs for one env, the registry entry
    exposes head/tail body indices that match add_rod_graph's return order."""
    import newton
    import warp as wp

    from isaaclab_contrib.cable.cable_object import CableRegistryEntry, add_cable_entry_to_builder

    builder = newton.ModelBuilder()
    entry = CableRegistryEntry(
        prim_path="/World/Cable001",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.05, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0)],
        edges=[(0, 1), (1, 2)],
        radius=0.005,
    )
    add_cable_entry_to_builder(
        builder,
        entry,
        env_idx=0,
        env_position=[0.0, 0.0, 0.0],
        env_rotation=[0.0, 0.0, 0.0, 1.0],
        cable_idx=0,
    )

    assert len(entry.head_segment_body_indices) == 1
    assert len(entry.tail_segment_body_indices) == 1
    # First edge body comes before last edge body in builder body order.
    assert entry.head_segment_body_indices[0] < entry.tail_segment_body_indices[0]
    # tail = head + (num_edges - 1) since add_rod_graph allocates one body per edge.
    assert entry.tail_segment_body_indices[0] - entry.head_segment_body_indices[0] == len(entry.edges) - 1


def test_pending_cable_attachments_is_initialized_by_install_hooks():
    """install_cable_builder_hooks() must reset _pending_cable_attachments to []."""
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    from isaaclab_contrib.cable.cable_object import install_cable_builder_hooks

    install_cable_builder_hooks()
    assert hasattr(SimulationManager, "_pending_cable_attachments")
    assert SimulationManager._pending_cable_attachments == []

    # Calling install again resets it to empty even if entries were appended.
    SimulationManager._pending_cable_attachments.append(("fake_entry",))
    install_cable_builder_hooks()
    assert SimulationManager._pending_cable_attachments == []


def test_apply_cable_attachments_adds_fixed_joint():
    """For one world: registering a cable and a plug, then running the cable
    hook followed by the attachment hook, must produce exactly one new fixed
    joint between the plug body and the cable's tail segment body."""
    import newton
    import warp as wp
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    from isaaclab_contrib.cable import CableAttachmentCfg
    from isaaclab_contrib.cable.cable_object import (
        CableRegistryEntry,
        add_cable_entry_to_builder,
        apply_cable_attachments_to_builder,
    )

    # Fresh registries.
    SimulationManager._cable_registry = []
    SimulationManager._pending_cable_attachments = []

    builder = newton.ModelBuilder()

    # Mirror the cloner: bodies and joints are added inside a begin/end world
    # block so they're tagged with ``body_world == 0`` — the attachment hook
    # filters by ``body_world`` to bind to the correct world's plug.
    builder.begin_world()

    # Plug rigid body added first, with body_label matching the target path.
    plug_path = "/World/Plug001"
    plug_idx = builder.add_body(xform=wp.transform_identity(), label=plug_path)
    builder.add_joint_free(child=plug_idx)

    # Cable.
    entry = CableRegistryEntry(
        prim_path="/World/Cable001",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.05, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0)],
        edges=[(0, 1), (1, 2)],
        radius=0.005,
    )
    SimulationManager._cable_registry.append(entry)
    SimulationManager._pending_cable_attachments.append(
        (0, CableAttachmentCfg(target_prim_path=plug_path, cable_anchor="tail"))
    )

    add_cable_entry_to_builder(
        builder,
        entry,
        env_idx=0,
        env_position=[0.0, 0.0, 0.0],
        env_rotation=[0.0, 0.0, 0.0, 1.0],
        cable_idx=0,
    )
    joints_after_cable = builder.joint_count

    apply_cable_attachments_to_builder(
        builder,
        world_idx=0,
        env_position=[0.0, 0.0, 0.0],
        env_rotation=[0.0, 0.0, 0.0, 1.0],
    )
    joints_after_attachment = builder.joint_count

    builder.end_world()

    # One new joint added by the attachment hook.
    assert joints_after_attachment - joints_after_cable == 1, (
        f"expected 1 new joint from attachment, got {joints_after_attachment - joints_after_cable}"
    )

    # That joint must connect the cable's tail body (parent) and plug_idx
    # (child). The current implementation calls ``add_joint_fixed`` with the
    # cable anchor as the ``parent`` argument and the target rigid body as the
    # ``child`` argument, so plug_idx lands on the ``joint_child`` column.
    new_joint_idx = joints_after_attachment - 1
    assert builder.joint_parent[new_joint_idx] == entry.tail_segment_body_indices[0]
    assert builder.joint_child[new_joint_idx] == plug_idx


def test_apply_cable_attachments_missing_target_raises():
    """If target_prim_path does not match any body_label, a ValueError must be
    raised that names the missing path and lists the available labels for the
    world being built."""
    import newton
    import warp as wp
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    from isaaclab_contrib.cable import CableAttachmentCfg
    from isaaclab_contrib.cable.cable_object import (
        CableRegistryEntry,
        add_cable_entry_to_builder,
        apply_cable_attachments_to_builder,
    )

    SimulationManager._cable_registry = []
    SimulationManager._pending_cable_attachments = []

    builder = newton.ModelBuilder()
    builder.begin_world()

    entry = CableRegistryEntry(
        prim_path="/World/Cable001",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.05, 0.0, 0.0)],
        edges=[(0, 1)],
        radius=0.005,
    )
    SimulationManager._cable_registry.append(entry)
    SimulationManager._pending_cable_attachments.append(
        (0, CableAttachmentCfg(target_prim_path="/World/DoesNotExist", cable_anchor="tail"))
    )

    add_cable_entry_to_builder(
        builder,
        entry,
        env_idx=0,
        env_position=[0.0, 0.0, 0.0],
        env_rotation=[0.0, 0.0, 0.0, 1.0],
        cable_idx=0,
    )

    with pytest.raises(ValueError, match=r"/World/DoesNotExist"):
        apply_cable_attachments_to_builder(
            builder,
            world_idx=0,
            env_position=[0.0, 0.0, 0.0],
            env_rotation=[0.0, 0.0, 0.0, 1.0],
        )
    builder.end_world()


def test_apply_cable_attachments_per_world_resolves_correct_plug():
    """Under multi-world cloning, the attachment hook must bind each world's
    cable to that world's plug — not env-0's copy. Regression for the
    body_label.index() bug that ignored body_world."""
    import newton
    import warp as wp

    from isaaclab_contrib.cable import CableAttachmentCfg
    from isaaclab_contrib.cable.cable_object import (
        CableRegistryEntry,
        add_cable_entry_to_builder,
        apply_cable_attachments_to_builder,
    )
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    # Fresh registries.
    SimulationManager._cable_registry = []
    SimulationManager._pending_cable_attachments = []

    builder = newton.ModelBuilder()

    # Register one cable that will be replicated into every world. The cable
    # hook appends per-world head/tail body indices to the entry as each world
    # is built.
    plug_path = "/World/Plug"
    entry = CableRegistryEntry(
        prim_path="/World/Cable",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.05, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0)],
        edges=[(0, 1), (1, 2)],
        radius=0.005,
    )
    SimulationManager._cable_registry.append(entry)
    SimulationManager._pending_cable_attachments.append(
        (0, CableAttachmentCfg(target_prim_path=plug_path, cable_anchor="tail"))
    )

    # Mirror the real cloner: per-world ``begin_world``/``end_world`` block adds
    # this env's plug AND runs the per-world builder hooks. Each plug uses the
    # SAME source label ``plug_path`` (this is what the cloner produces before
    # ``_rename_builder_labels`` runs); the hook must filter by ``body_world``
    # to bind to *this* world's plug, not env-0's.
    plug_indices_by_world: list[int] = []
    for world_idx in range(2):
        builder.begin_world()
        plug_idx = builder.add_body(xform=wp.transform_identity(), label=plug_path)
        builder.add_joint_free(child=plug_idx)
        plug_indices_by_world.append(plug_idx)

        add_cable_entry_to_builder(
            builder, entry, env_idx=world_idx,
            env_position=[0.0, 0.0, 0.0], env_rotation=[0.0, 0.0, 0.0, 1.0],
            cable_idx=0,
        )
        joints_before = builder.joint_count
        apply_cable_attachments_to_builder(
            builder, world_idx=world_idx,
            env_position=[0.0, 0.0, 0.0], env_rotation=[0.0, 0.0, 0.0, 1.0],
        )
        joints_after = builder.joint_count
        builder.end_world()

        assert joints_after - joints_before == 1, (
            f"world {world_idx}: expected 1 new joint, got {joints_after - joints_before}"
        )
        new_joint_idx = joints_after - 1
        # The attachment joint must reference THIS world's plug — not env 0's.
        # The current implementation calls ``add_joint_fixed(parent=cable, child=plug)``,
        # so the plug ends up on the ``joint_child`` column.
        assert builder.joint_child[new_joint_idx] == plug_indices_by_world[world_idx], (
            f"world {world_idx}: joint child {builder.joint_child[new_joint_idx]} "
            f"!= this-world plug {plug_indices_by_world[world_idx]} "
            f"(env-0 plug was {plug_indices_by_world[0]})"
        )
        assert builder.joint_parent[new_joint_idx] == entry.tail_segment_body_indices[world_idx]


def _build_cable_plug_scene(
    plug_kinematic: bool,
    sim_dt: float = 0.01,
    rigid_body_contact_buffer_size: int = 64,
    num_substeps: int = 4,
    rigid_contact_k_start: float = 1.0e2,
    cable_stretch_stiffness: float = 1e6,
    cable_stretch_damping: float = 1e-4,
    shape_material_ke: float | None = None,
    shape_material_kd: float | None = None,
    shape_material_mu: float | None = None,
    cable_anchor: Literal["head", "tail"] = "tail",
):
    """Shared scaffolding: spawn a ground, a cable, a plug, and weld the cable's
    selected anchor (head or tail) to the plug.

    Returns (sim, cable, plug, plug_world_pos_initial)."""
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    from isaaclab_contrib.cable import CableAttachmentCfg, CableObject, CableObjectCfg
    from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

    physics_cfg = NewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=20,
            rigid_body_contact_buffer_size=rigid_body_contact_buffer_size,
            rigid_contact_k_start=rigid_contact_k_start,
        ),
        num_substeps=num_substeps,
    )
    model_cfg_kwargs = {}
    if shape_material_ke is not None:
        model_cfg_kwargs["shape_material_ke"] = shape_material_ke
    if shape_material_kd is not None:
        model_cfg_kwargs["shape_material_kd"] = shape_material_kd
    if shape_material_mu is not None:
        model_cfg_kwargs["shape_material_mu"] = shape_material_mu
    physics_cfg.model_cfg = NewtonModelCfg(**model_cfg_kwargs)
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, physics=physics_cfg)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    plug_world_pos = (0.0, 0.0, 1.0)
    plug_world_quat = (1.0, 0.0, 0.0, 0.0)
    plug_cfg = RigidObjectCfg(
        prim_path="/World/Plug",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/mmichelis/Documents/IsaacLab-Origin/scripts/demos/plug_mesh_flange_only.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=plug_kinematic),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=plug_world_pos, rot=plug_world_quat),
    )
    plug = RigidObject(cfg=plug_cfg)

    num_points = 10
    seg_len = 0.02
    if cable_anchor == "tail":
        # Place cable so its *tail body* (one segment back from the last node)
        # co-locates with the plug at spawn. The last edge body sits at the
        # midpoint of segment (num_points - 2, num_points - 1) along local +X,
        # i.e. ``(num_points - 1.5) * seg_len`` from the cable origin. Using
        # ``(num_points - 2) * seg_len`` puts the *tail node* one segment past
        # the plug and the *tail body* at the plug, avoiding a step-1 snap.
        cable_init_pos = (
            plug_world_pos[0] - (num_points - 2) * seg_len,
            plug_world_pos[1],
            plug_world_pos[2],
        )
    elif cable_anchor == "head":
        # For "head" anchoring the cable's first edge body sits at node 0 =
        # ``init_state.pos``, so co-locate the cable origin with the plug.
        cable_init_pos = plug_world_pos
    else:
        raise ValueError(f"cable_anchor must be 'head' or 'tail', got {cable_anchor!r}")
    cable_cfg = CableObjectCfg(
        prim_path="/World/Cable",
        spawn=sim_utils.CableCfg(
            positions=[(i * seg_len, 0.0, 0.0) for i in range(num_points)],
            width=0.006,
            physics_material=NewtonCableMaterialCfg(
                stretch_stiffness=cable_stretch_stiffness,
                bend_stiffness=1e-4,
                stretch_damping=cable_stretch_damping,
                bend_damping=1e-4,
                density=100.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=CableObjectCfg.InitialStateCfg(
            pos=cable_init_pos,
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        attachments=[
            CableAttachmentCfg(
                target_prim_path="/World/Plug",  # may need adjustment - see notes
                cable_anchor=cable_anchor,
                cable_local_pos=(0.0, 0.0, 0.0),
                cable_local_quat=(0.0, 0.0, 0.0, 1.0),
            ),
        ],
    )
    cable = CableObject(cfg=cable_cfg)

    sim.reset()
    return sim, cable, plug, plug_world_pos


def test_cable_tail_tracks_kinematic_plug():
    """With the plug pinned (kinematic), the cable's tail world position must
    stay glued to the plug's pose to within 1 mm after 200 steps under gravity."""
    sim, cable, plug, plug_pos0 = _build_cable_plug_scene(plug_kinematic=True)

    sim_dt = sim.get_physics_dt()
    for _ in range(200):
        sim.step()
        cable.update(sim_dt)
        plug.update(sim_dt)

    # Plug stayed put.
    plug_pos_now = plug.data.root_pos_w.torch[0].cpu().numpy()
    assert abs(plug_pos_now[0] - plug_pos0[0]) < 1e-3
    assert abs(plug_pos_now[1] - plug_pos0[1]) < 1e-3
    assert abs(plug_pos_now[2] - plug_pos0[2]) < 1e-3

    # Cable tail (last rod-segment body) tracks the plug. Read the tail body's
    # world transform directly from Newton's ``state_0.body_q`` via the body
    # index recorded on the cable registry entry, since :class:`CableObject`'s
    # ArticulationData does not expose per-rod-segment ``body_pos_w`` reliably
    # (cable joints are not first-class in :class:`ArticulationView`).
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    body_q = SimulationManager._state_0.body_q.numpy()
    tail_body_idx = cable._registry_entry.tail_segment_body_indices[0]
    tail_pos = body_q[tail_body_idx, 0:3]
    assert (
        (tail_pos[0] - plug_pos_now[0]) ** 2
        + (tail_pos[1] - plug_pos_now[1]) ** 2
        + (tail_pos[2] - plug_pos_now[2]) ** 2
    ) ** 0.5 < 1e-3, f"cable tail {tail_pos} did not track kinematic plug {plug_pos_now}"
    type(sim).clear_instance()


def test_cable_tail_tracks_falling_plug():
    """With the plug dynamic (not pinned), gravity drags the plug down and
    the cable's tail must remain welded to it each step (within tolerance).

    Uses a softer cable (``stretch_stiffness=1e3``) and softer body-shape
    contact (``shape_material_ke=1e3``) than the kinematic test. The 1e6 stretch
    stiffness used in the kinematic test causes the cable-plug system to
    explode here because the cable is light (~70 mg per body) so the
    cable/joint natural frequency is much higher than the substep rate. The
    softer cable parameters keep the simulation stable through ground contact
    while still tracking the plug to within a few millimetres.
    """
    sim, cable, plug, plug_pos0 = _build_cable_plug_scene(
        plug_kinematic=False,
        rigid_body_contact_buffer_size=1024,
        num_substeps=8,
        rigid_contact_k_start=1.0e1,
        cable_stretch_stiffness=1e3,
        cable_stretch_damping=1e-1,
        shape_material_ke=1.0e3,
        shape_material_kd=1.0e0,
        shape_material_mu=1.0,
    )

    from isaaclab_newton.physics import NewtonManager as SimulationManager

    # The plug's Newton body index. The plug is the first body added to the
    # builder in our scene (cable bodies come next). :class:`RigidObject` does
    # not propagate Newton's live ``body_q`` back into its cached
    # ``root_pos_w`` for objects that aren't part of an articulation, so we
    # read the plug pose directly from ``_state_0.body_q`` -- the same source
    # we use for the cable tail.
    plug_body_idx = 0

    sim_dt = sim.get_physics_dt()
    max_err = 0.0
    plug_pos_now = None
    for _ in range(200):
        sim.step()
        cable.update(sim_dt)
        plug.update(sim_dt)

        body_q = SimulationManager._state_0.body_q.numpy()
        plug_pos_now = body_q[plug_body_idx, 0:3]
        tail_body_idx = cable._registry_entry.tail_segment_body_indices[0]
        tail_pos = body_q[tail_body_idx, 0:3]
        err = (
            (tail_pos[0] - plug_pos_now[0]) ** 2
            + (tail_pos[1] - plug_pos_now[1]) ** 2
            + (tail_pos[2] - plug_pos_now[2]) ** 2
        ) ** 0.5
        max_err = max(max_err, err)

    assert plug_pos_now[2] < plug_pos0[2] - 0.05, (
        f"plug did not fall under gravity: started {plug_pos0[2]}, ended {plug_pos_now[2]}"
    )
    # The 1.5 cm tolerance accommodates the brief transient when the cable
    # drapes onto the ground at ~step 50 (observed peak error ~9 mm). At rest
    # the tail-plug error settles to well under 1 mm.
    assert max_err < 1.5e-2, f"cable tail drifted from plug; max error {max_err} m"
    type(sim).clear_instance()


def test_cable_head_anchor_welds_first_segment():
    """cable_anchor="head" must weld the cable's first rod-segment body to the
    plug, not the last one. Tail should hang free under gravity."""
    sim, cable, plug, plug_pos0 = _build_cable_plug_scene(
        plug_kinematic=True,
        cable_anchor="head",
    )

    from isaaclab_newton.physics import NewtonManager as SimulationManager

    sim_dt = sim.get_physics_dt()
    for _ in range(200):
        sim.step()
        cable.update(sim_dt)
        plug.update(sim_dt)

    body_q = SimulationManager._state_0.body_q.numpy()
    head_body_idx = cable._registry_entry.head_segment_body_indices[0]
    tail_body_idx = cable._registry_entry.tail_segment_body_indices[0]
    head_pos = body_q[head_body_idx, 0:3]
    tail_pos = body_q[tail_body_idx, 0:3]
    plug_pos_now = body_q[0, 0:3]  # plug is the first body added to the builder

    head_err = (
        (head_pos[0] - plug_pos_now[0]) ** 2
        + (head_pos[1] - plug_pos_now[1]) ** 2
        + (head_pos[2] - plug_pos_now[2]) ** 2
    ) ** 0.5
    tail_err = (
        (tail_pos[0] - plug_pos_now[0]) ** 2
        + (tail_pos[1] - plug_pos_now[1]) ** 2
        + (tail_pos[2] - plug_pos_now[2]) ** 2
    ) ** 0.5

    assert head_err < 1e-3, f"head should track plug: head_err {head_err} m, head_pos {head_pos}, plug {plug_pos_now}"
    # Tail should hang freely well below the plug under gravity.
    assert tail_err > 0.05, (
        f"tail should not track plug for head anchor: tail_err {tail_err} m, tail_pos {tail_pos}, plug {plug_pos_now}"
    )
    type(sim).clear_instance()


def test_cable_with_head_and_tail_attachments_forms_catenary():
    """Pin BOTH ends of one cable to two separate kinematic plugs spaced apart.
    Both endpoints must stay welded; the middle should sag below the endpoints."""
    from isaaclab_newton.physics import NewtonCfg
    from isaaclab_newton.physics import NewtonManager as SimulationManager
    from isaaclab_newton.sim.spawners.materials import NewtonCableMaterialCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    from isaaclab_contrib.cable import CableAttachmentCfg, CableObject, CableObjectCfg
    from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

    physics_cfg = NewtonCfg(
        solver_cfg=VBDSolverCfg(iterations=30, rigid_body_contact_buffer_size=1024),
        num_substeps=8,
    )
    physics_cfg.model_cfg = NewtonModelCfg()
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, physics=physics_cfg)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())

    plug_a_pos = (0.0, 0.0, 1.0)
    plug_b_pos = (0.3, 0.0, 1.0)
    plug_a = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/PlugA",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/mmichelis/Documents/IsaacLab-Origin/scripts/demos/plug_mesh_flange_only.usda",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=plug_a_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        )
    )
    plug_b = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/PlugB",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/home/mmichelis/Documents/IsaacLab-Origin/scripts/demos/plug_mesh_flange_only.usda",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=plug_b_pos, rot=(1.0, 0.0, 0.0, 0.0)),
        )
    )

    # Span more cable than plug separation (0.3 m) so it sags. Cable head at plug A,
    # tail body co-located with plug B accounting for the head body offset.
    num_points = 25
    seg_len = 0.02  # total length 0.48 m > 0.3 m separation
    # Cable head body is at node 0 = init_state.pos. Tail body is at node N-2 in local coords.
    # For an unstretched cable laid out along +X with init at plug A, the tail body's local
    # position is ((N-2)*seg_len, 0, 0). For both ends to align at spawn we'd need plug B
    # at distance (N-2)*seg_len = 0.46 m. The plug B is closer (0.3 m), so the joint pulls
    # the cable into a catenary. Initial-snap from the joint is up to 0.16 m along X.
    cable = CableObject(
        cfg=CableObjectCfg(
            prim_path="/World/Cable",
            spawn=sim_utils.CableCfg(
                positions=[(i * seg_len, 0.0, 0.0) for i in range(num_points)],
                width=0.006,
                physics_material=NewtonCableMaterialCfg(
                    stretch_stiffness=1e3,
                    bend_stiffness=1e-4,
                    stretch_damping=1e-1,
                    bend_damping=1e-4,
                    density=100.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=plug_a_pos, rot=(0.0, 0.0, 0.0, 1.0)),
            attachments=[
                CableAttachmentCfg(target_prim_path="/World/PlugA", cable_anchor="head"),
                CableAttachmentCfg(target_prim_path="/World/PlugB", cable_anchor="tail"),
            ],
        )
    )

    sim.reset()

    sim_dt = sim.get_physics_dt()
    for _ in range(300):
        sim.step()
        cable.update(sim_dt)
        plug_a.update(sim_dt)
        plug_b.update(sim_dt)

    body_q = SimulationManager._state_0.body_q.numpy()
    head_body_idx = cable._registry_entry.head_segment_body_indices[0]
    tail_body_idx = cable._registry_entry.tail_segment_body_indices[0]
    # Resolve plug body indices by label so the test isn't fragile to
    # insertion-order changes. Newton stores the per-body label as
    # ``body_label`` on the live ``Model`` (and on the ``ModelBuilder``).
    model = SimulationManager._model
    body_label_attr = "body_label" if hasattr(model, "body_label") else "body_key"
    body_label = list(getattr(model, body_label_attr))
    pa_body_idx = body_label.index("/World/PlugA")
    pb_body_idx = body_label.index("/World/PlugB")
    head_pos = body_q[head_body_idx, 0:3]
    tail_pos = body_q[tail_body_idx, 0:3]
    pa = body_q[pa_body_idx, 0:3]
    pb = body_q[pb_body_idx, 0:3]

    # Middle of the cable: choose the middle EDGE body.
    num_edges = num_points - 1
    mid_edge_body_idx = head_body_idx + num_edges // 2
    mid = body_q[mid_edge_body_idx, 0:3]

    head_err = ((head_pos[0] - pa[0]) ** 2 + (head_pos[1] - pa[1]) ** 2 + (head_pos[2] - pa[2]) ** 2) ** 0.5
    tail_err = ((tail_pos[0] - pb[0]) ** 2 + (tail_pos[1] - pb[1]) ** 2 + (tail_pos[2] - pb[2]) ** 2) ** 0.5

    assert head_err < 1.5e-2, f"head not at plug A: head_pos {head_pos}, plug_a {pa}, err {head_err}"
    assert tail_err < 1.5e-2, f"tail not at plug B: tail_pos {tail_pos}, plug_b {pb}, err {tail_err}"
    # Middle sags below endpoints (gravity is -Z). With the soft stretch
    # stiffness used here (1e3 N/m), the cable stretches enough that the
    # observed sag is ~2-3 cm rather than the analytical catenary depth for
    # an inextensible 0.48 m cable across 0.30 m. The band is set tight
    # enough to catch a regression to "no sag" while tolerating solver noise.
    sag = min(pa[2], pb[2]) - mid[2]
    assert 0.015 < sag < 0.20, (
        f"middle did not sag in expected range: mid {mid}, plugA {pa}, plugB {pb}, sag {sag}"
    )
    # The catenary middle (chosen as the geometric middle edge body) should
    # land between the two plugs in X. The cable stretches non-uniformly
    # under the soft stretch stiffness, so we allow generous slack while
    # still excluding the failure mode where the cable collapses to one plug.
    midpoint_x = 0.5 * (pa[0] + pb[0])
    assert abs(mid[0] - midpoint_x) < 0.15, (
        f"catenary middle not roughly centered: mid_x {mid[0]}, expected ~{midpoint_x}"
    )

    type(sim).clear_instance()
