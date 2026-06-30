# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for cable-endpoint ↔ rigid-body attachments."""

from __future__ import annotations

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest


def test_cable_attachment_cfg_defaults_and_types():
    """CableAttachmentCfg accepts head/tail anchors and exposes the documented defaults."""
    from isaaclab_contrib.cable import CableAttachmentCfg

    cfg = CableAttachmentCfg(target_prim_path="/World/Plug001", cable_anchor=-1)
    assert cfg.target_prim_path == "/World/Plug001"
    assert cfg.cable_anchor == -1
    assert cfg.cable_local_pos == (0.0, 0.0, 0.0)
    assert cfg.cable_local_quat == (0.0, 0.0, 0.0, 1.0)
    assert cfg.target_local_pos == (0.0, 0.0, 0.0)
    assert cfg.target_local_quat == (0.0, 0.0, 0.0, 1.0)

    cfg2 = CableAttachmentCfg(
        target_prim_path="/Foo",
        cable_anchor=0,
        cable_local_pos=(1.0, 2.0, 3.0),
        cable_local_quat=(0.5, 0.5, 0.5, 0.5),
        target_local_pos=(4.0, 5.0, 6.0),
        target_local_quat=(0.7071, 0.7071, 0.0, 0.0),
    )
    assert cfg2.cable_anchor == 0
    assert cfg2.cable_local_pos == (1.0, 2.0, 3.0)
    assert cfg2.cable_local_quat == (0.5, 0.5, 0.5, 0.5)
    assert cfg2.target_local_pos == (4.0, 5.0, 6.0)
    assert cfg2.target_local_quat == (0.7071, 0.7071, 0.0, 0.0)


def test_cable_object_cfg_attachments_field_default_empty():
    """CableObjectCfg exposes an `attachments` list field that defaults to empty."""
    from isaaclab_contrib.cable import CableAttachmentCfg
    from isaaclab_contrib.cable.cable_object_cfg import CableObjectCfg

    cfg = CableObjectCfg(prim_path="/World/Cable001")
    assert hasattr(cfg, "attachments")
    assert cfg.attachments == []

    cfg2 = CableObjectCfg(
        prim_path="/World/Cable001",
        attachments=[CableAttachmentCfg(target_prim_path="/World/Plug001", cable_anchor=-1)],
    )
    assert len(cfg2.attachments) == 1
    assert cfg2.attachments[0].target_prim_path == "/World/Plug001"
    assert cfg2.attachments[0].cable_anchor == -1


def test_cable_registry_records_head_tail_body_indices():
    """Registry entry exposes head/tail body indices matching add_rod_graph's return order."""
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

    assert len(entry.segment_body_indices) == 1
    assert len(entry.segment_body_indices[0]) == len(entry.edges)
    assert entry.segment_body_indices[0][0] < entry.segment_body_indices[0][-1]
    assert entry.segment_body_indices[0][-1] - entry.segment_body_indices[0][0] == len(entry.edges) - 1


def test_pending_cable_attachments_is_initialized_by_install_hooks():
    """install_cable_builder_hooks() must reset _pending_cable_attachments to []."""
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    from isaaclab_contrib.cable.cable_object import install_cable_builder_hooks

    install_cable_builder_hooks()
    assert hasattr(SimulationManager, "_pending_cable_attachments")
    assert SimulationManager._pending_cable_attachments == []

    SimulationManager._pending_cable_attachments.append(("fake_entry",))
    install_cable_builder_hooks()
    assert SimulationManager._pending_cable_attachments == []


def test_apply_cable_attachments_adds_fixed_joint():
    """One cable + one plug produces exactly one fixed joint between plug and tail segment."""
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

    plug_path = "/World/Plug001"
    plug_idx = builder.add_body(xform=wp.transform_identity(), label=plug_path)
    builder.add_joint_free(child=plug_idx)

    entry = CableRegistryEntry(
        prim_path="/World/Cable001",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.05, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0)],
        edges=[(0, 1), (1, 2)],
        radius=0.005,
    )
    SimulationManager._cable_registry.append(entry)
    SimulationManager._pending_cable_attachments.append(
        (0, CableAttachmentCfg(target_prim_path=plug_path, cable_anchor=-1))
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

    assert joints_after_attachment - joints_after_cable == 1

    # add_joint_fixed is called with cable anchor as parent, target as child.
    new_joint_idx = joints_after_attachment - 1
    assert builder.joint_parent[new_joint_idx] == entry.segment_body_indices[0][-1]
    assert builder.joint_child[new_joint_idx] == plug_idx


def test_apply_cable_attachments_missing_target_raises():
    """Unknown target_prim_path raises ValueError naming the missing path."""
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
        (0, CableAttachmentCfg(target_prim_path="/World/DoesNotExist", cable_anchor=-1))
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
    """Each world binds to its own plug, not env-0's (regression for body_label.index() ignoring body_world)."""
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

    plug_path = "/World/Plug"
    entry = CableRegistryEntry(
        prim_path="/World/Cable",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.05, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0)],
        edges=[(0, 1), (1, 2)],
        radius=0.005,
    )
    SimulationManager._cable_registry.append(entry)
    SimulationManager._pending_cable_attachments.append(
        (0, CableAttachmentCfg(target_prim_path=plug_path, cable_anchor=-1))
    )

    # Each plug uses the same source label (pre-_rename_builder_labels state);
    # the hook must filter by body_world.
    plug_indices_by_world: list[int] = []
    for world_idx in range(2):
        builder.begin_world()
        plug_idx = builder.add_body(xform=wp.transform_identity(), label=plug_path)
        builder.add_joint_free(child=plug_idx)
        plug_indices_by_world.append(plug_idx)

        add_cable_entry_to_builder(
            builder,
            entry,
            env_idx=world_idx,
            env_position=[0.0, 0.0, 0.0],
            env_rotation=[0.0, 0.0, 0.0, 1.0],
            cable_idx=0,
        )
        joints_before = builder.joint_count
        apply_cable_attachments_to_builder(
            builder,
            world_idx=world_idx,
            env_position=[0.0, 0.0, 0.0],
            env_rotation=[0.0, 0.0, 0.0, 1.0],
        )
        joints_after = builder.joint_count
        builder.end_world()

        assert joints_after - joints_before == 1
        new_joint_idx = joints_after - 1
        assert builder.joint_child[new_joint_idx] == plug_indices_by_world[world_idx], (
            f"world {world_idx}: bound to env-0 plug instead of own"
        )
        assert builder.joint_parent[new_joint_idx] == entry.segment_body_indices[world_idx][-1]


def test_cable_labels_and_attachments_expand_env_regex_under_cloning():
    """env_.* tokens in cable/target prim paths are pre-expanded per-world.

    Matches both unexpanded (USD) and expanded (builder-hook) labels.
    """
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

    # cable_a → USD plug (label kept as unexpanded regex template).
    # cable_b → builder-hook anchor (label pre-expanded to env_<N>).
    cable_a = CableRegistryEntry(
        prim_path="/World/envs/env_.*/CableA",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.05, 0.0, 0.0), wp.vec3(0.1, 0.0, 0.0)],
        edges=[(0, 1), (1, 2)],
        radius=0.005,
    )
    cable_b = CableRegistryEntry(
        prim_path="/World/envs/env_.*/CableB",
        node_positions=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.05, 0.0, 0.0)],
        edges=[(0, 1)],
        radius=0.005,
    )
    SimulationManager._cable_registry.append(cable_a)
    SimulationManager._cable_registry.append(cable_b)

    plug_regex_path = "/World/envs/env_.*/Plug"
    anchor_regex_path = "/World/envs/env_.*/Anchor"
    SimulationManager._pending_cable_attachments.append(
        (0, CableAttachmentCfg(target_prim_path=plug_regex_path, cable_anchor=-1))
    )
    SimulationManager._pending_cable_attachments.append(
        (1, CableAttachmentCfg(target_prim_path=anchor_regex_path, cable_anchor=0))
    )

    plug_indices_by_world: list[int] = []
    anchor_indices_by_world: list[int] = []
    for world_idx in range(2):
        builder.begin_world()

        plug_idx = builder.add_body(xform=wp.transform_identity(), label=plug_regex_path)
        builder.add_joint_free(child=plug_idx)
        plug_indices_by_world.append(plug_idx)

        anchor_label = anchor_regex_path.replace("env_.*", f"env_{world_idx}")
        anchor_idx = builder.add_body(xform=wp.transform_identity(), label=anchor_label)
        builder.add_joint_free(child=anchor_idx)
        anchor_indices_by_world.append(anchor_idx)

        add_cable_entry_to_builder(
            builder,
            cable_a,
            env_idx=world_idx,
            env_position=[0.0, 0.0, 0.0],
            env_rotation=[0.0, 0.0, 0.0, 1.0],
            cable_idx=0,
        )
        add_cable_entry_to_builder(
            builder,
            cable_b,
            env_idx=world_idx,
            env_position=[0.0, 0.0, 0.0],
            env_rotation=[0.0, 0.0, 0.0, 1.0],
            cable_idx=1,
        )
        joints_before = builder.joint_count
        apply_cable_attachments_to_builder(
            builder,
            world_idx=world_idx,
            env_position=[0.0, 0.0, 0.0],
            env_rotation=[0.0, 0.0, 0.0, 1.0],
        )
        joints_after = builder.joint_count
        builder.end_world()

        assert joints_after - joints_before == 2

        # Cloner doesn't rewrite builder-hook bodies, so cable rod labels must already be env-expanded.
        cable_a_tail_idx = cable_a.segment_body_indices[world_idx][-1]
        expected_a_label = f"/World/envs/env_{world_idx}/CableA/cable_edge_body_{len(cable_a.edges) - 1}"
        assert builder.body_label[cable_a_tail_idx] == expected_a_label
        assert builder.body_world[cable_a_tail_idx] == world_idx

        attach_a_idx = joints_after - 2
        assert builder.joint_child[attach_a_idx] == plug_indices_by_world[world_idx]
        assert builder.joint_parent[attach_a_idx] == cable_a.segment_body_indices[world_idx][-1]

        attach_b_idx = joints_after - 1
        assert builder.joint_child[attach_b_idx] == anchor_indices_by_world[world_idx]
        assert builder.joint_parent[attach_b_idx] == cable_b.segment_body_indices[world_idx][0]


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
    cable_anchor: int = -1,
):
    """Spawn ground + cable + plug and weld the selected cable segment to the plug.

    Returns (sim, cable, plug, plug_world_pos_initial). cable_anchor is -1 (tail) or 0 (head).
    """
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
    if cable_anchor == -1:
        # Last edge body sits at midpoint of segment (N-2, N-1), i.e. (N-1.5)*seg_len from origin.
        # Using (N-2)*seg_len puts the tail body at the plug (tail node one segment past), avoiding a step-1 snap.
        cable_init_pos = (
            plug_world_pos[0] - (num_points - 2) * seg_len,
            plug_world_pos[1],
            plug_world_pos[2],
        )
    elif cable_anchor == 0:
        cable_init_pos = plug_world_pos
    else:
        raise ValueError(f"_build_cable_plug_scene only supports cable_anchor in {{-1, 0}}, got {cable_anchor!r}")
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
                target_prim_path="/World/Plug",
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
    """Cable tail stays glued to a pinned plug within 1 mm after 200 steps."""
    sim, cable, plug, plug_pos0 = _build_cable_plug_scene(plug_kinematic=True)

    sim_dt = sim.get_physics_dt()
    for _ in range(200):
        sim.step()
        cable.update(sim_dt)
        plug.update(sim_dt)

    plug_pos_now = plug.data.root_pos_w.torch[0].cpu().numpy()
    assert abs(plug_pos_now[0] - plug_pos0[0]) < 1e-3
    assert abs(plug_pos_now[1] - plug_pos0[1]) < 1e-3
    assert abs(plug_pos_now[2] - plug_pos0[2]) < 1e-3

    # Read tail pose from state_0.body_q: CableObject's ArticulationData doesn't expose
    # per-rod-segment body_pos_w reliably (cable joints aren't first-class in ArticulationView).
    from isaaclab_newton.physics import NewtonManager as SimulationManager

    body_q = SimulationManager._state_0.body_q.numpy()
    tail_body_idx = cable._registry_entry.segment_body_indices[0][-1]
    tail_pos = body_q[tail_body_idx, 0:3]
    assert (
        (tail_pos[0] - plug_pos_now[0]) ** 2
        + (tail_pos[1] - plug_pos_now[1]) ** 2
        + (tail_pos[2] - plug_pos_now[2]) ** 2
    ) ** 0.5 < 1e-3, f"cable tail {tail_pos} did not track kinematic plug {plug_pos_now}"
    type(sim).clear_instance()


def test_cable_tail_tracks_falling_plug():
    """Tail remains welded to a dynamic falling plug within a few mm."""
    # Softer cable/contact params chosen to keep the cable-plug system stable through
    # ground contact (1e6 stretch stiffness explodes here because cable bodies are ~70 mg).
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

    # Read plug pose from state_0.body_q: RigidObject.root_pos_w isn't updated from
    # Newton's live body_q for non-articulated objects.
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
        tail_body_idx = cable._registry_entry.segment_body_indices[0][-1]
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
    # 1.5 cm tolerance accommodates the transient when the cable drapes onto the ground
    # (~step 50, peak ~9 mm); at rest the error settles well under 1 mm.
    assert max_err < 1.5e-2, f"cable tail drifted from plug; max error {max_err} m"
    type(sim).clear_instance()


def test_cable_head_anchor_welds_first_segment():
    """cable_anchor=0 welds the first rod segment to the plug; tail hangs free."""
    sim, cable, plug, plug_pos0 = _build_cable_plug_scene(
        plug_kinematic=True,
        cable_anchor=0,
    )

    from isaaclab_newton.physics import NewtonManager as SimulationManager

    sim_dt = sim.get_physics_dt()
    for _ in range(200):
        sim.step()
        cable.update(sim_dt)
        plug.update(sim_dt)

    body_q = SimulationManager._state_0.body_q.numpy()
    head_body_idx = cable._registry_entry.segment_body_indices[0][0]
    tail_body_idx = cable._registry_entry.segment_body_indices[0][-1]
    head_pos = body_q[head_body_idx, 0:3]
    tail_pos = body_q[tail_body_idx, 0:3]
    plug_pos_now = body_q[0, 0:3]

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
    assert tail_err > 0.05, (
        f"tail should not track plug for head anchor: tail_err {tail_err} m, tail_pos {tail_pos}, plug {plug_pos_now}"
    )
    type(sim).clear_instance()


def test_cable_with_head_and_tail_attachments_forms_catenary():
    """Both endpoints pinned to separate plugs; cable middle sags below the line."""
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

    # 0.36 m cable across 0.30 m plug separation: modest slack drapes into a shallow catenary.
    num_points = 18
    seg_len = 0.02
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
                CableAttachmentCfg(target_prim_path="/World/PlugA", cable_anchor=0),
                CableAttachmentCfg(target_prim_path="/World/PlugB", cable_anchor=-1),
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
    head_body_idx = cable._registry_entry.segment_body_indices[0][0]
    tail_body_idx = cable._registry_entry.segment_body_indices[0][-1]
    # Resolve plug body indices by label to avoid coupling to insertion order.
    model = SimulationManager._model
    body_label_attr = "body_label" if hasattr(model, "body_label") else "body_key"
    body_label = list(getattr(model, body_label_attr))
    pa_body_idx = body_label.index("/World/PlugA")
    pb_body_idx = body_label.index("/World/PlugB")
    head_pos = body_q[head_body_idx, 0:3]
    tail_pos = body_q[tail_body_idx, 0:3]
    pa = body_q[pa_body_idx, 0:3]
    pb = body_q[pb_body_idx, 0:3]

    # With 0.06 m of slack the cable drapes into a symmetric shallow catenary whose
    # lowest body is the geometric (index) midpoint. Measure sag there, honestly.
    seg_indices = cable._registry_entry.segment_body_indices[0]
    cable_z = body_q[seg_indices, 2]
    mid_local = len(seg_indices) // 2
    mid_body_idx = seg_indices[mid_local]
    mid = body_q[mid_body_idx, 0:3]
    deepest_local = int(cable_z.argmin())
    deepest = body_q[seg_indices[deepest_local], 0:3]

    head_err = ((head_pos[0] - pa[0]) ** 2 + (head_pos[1] - pa[1]) ** 2 + (head_pos[2] - pa[2]) ** 2) ** 0.5
    tail_err = ((tail_pos[0] - pb[0]) ** 2 + (tail_pos[1] - pb[1]) ** 2 + (tail_pos[2] - pb[2]) ** 2) ** 0.5

    # 1.5 cm endpoint tolerance: the soft 1e3 N/m stretch stiffness allows visible stretching.
    assert head_err < 1.5e-2, f"head not at plug A: head_pos {head_pos}, plug_a {pa}, err {head_err}"
    assert tail_err < 1.5e-2, f"tail not at plug B: tail_pos {tail_pos}, plug_b {pb}, err {tail_err}"
    # Sag measured at the index midpoint, where a true catenary bottoms out (~5 cm here).
    sag = min(pa[2], pb[2]) - mid[2]
    assert 0.015 < sag < 0.20, f"cable did not sag in expected range: mid {mid}, plugA {pa}, plugB {pb}, sag {sag}"
    # No body may rise above the line between the pinned endpoints (rules out buckling/piling).
    z_line = min(pa[2], pb[2])
    assert cable_z.max() <= z_line + 1e-3, (
        f"a cable body rose above the endpoint line: max_z {cable_z.max()}, line {z_line}"
    )
    # The lowest body must sit strictly inside the plug span, centered (rules out a fold past a plug).
    lo_x, hi_x = sorted((pa[0], pb[0]))
    assert lo_x < deepest[0] < hi_x, (
        f"cable deepest point not within plug span: deepest_x {deepest[0]}, span [{lo_x}, {hi_x}]"
    )
    assert abs(deepest[0] - 0.5 * (lo_x + hi_x)) < 0.1, (
        f"cable deepest point not centered: deepest_x {deepest[0]}, center {0.5 * (lo_x + hi_x)}"
    )

    type(sim).clear_instance()
