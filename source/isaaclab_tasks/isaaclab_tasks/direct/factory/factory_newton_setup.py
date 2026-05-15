# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory: procedural Newton-only post-load asset setup.

Three entry points called from :class:`FactoryEnv`:

* :func:`apply_cfg_overrides` — runs before ``super().__init__``. Raises
  the physics rate, disables the OSC null-space term, wraps joint-less
  Factory assets as :class:`RigidObjectCfg`, tunes the arm and finger
  actuators, monkey-patches the cloner, and sizes the contact buffer.

* :func:`register_model_init_callback` — registers a Newton
  ``MODEL_INIT`` callback that runs before model finalization. Overrides
  joint target mode + ctrl-source on the Franka, filters base ↔ table
  contacts, retunes the nut/bolt contact materials, and builds per-shape
  hydroelastic SDFs.

* :func:`warm_up_kernels` — runs after ``_init_tensors``. Pre-JITs the
  Newton kernels the reset IK loop will hit.

PhysX runs never import this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import newton

from isaaclab.assets import ArticulationCfg, RigidObjectCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .factory_env import FactoryEnv
    from .factory_env_cfg import FactoryEnvCfg

_ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
_FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
# Substring matched against ``builder.body_label`` to find every body
# belonging to the Franka under the IsaacLab USD layout.
_ROBOT_BODY_PATH_SUBSTR = "/Robot/"


def register_model_init_callback() -> None:
    """Wire a Newton MODEL_INIT callback that mutates the builder."""
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.physics import PhysicsEvent

    NewtonManager.register_callback(
        lambda _ev: _model_init_callback(), PhysicsEvent.MODEL_INIT, name="factory_newton_setup"
    )


def apply_cfg_overrides(cfg: FactoryEnvCfg) -> None:
    """Apply Newton-only cfg overrides in place, *before* ``super().__init__``.

    Keeps the physics rate at 120 Hz with ``decimation = 8`` to keep the
    control rate at 15 Hz, disables the OSC null-space term, wraps joint-less
    Factory assets as :class:`RigidObjectCfg` (kinematic for static targets),
    bumps arm armature, zeroes arm damping, softens finger PD, monkey-patches
    the cloner to skip convex-hull simplification (so nut/bolt thread + hex
    SDFs survive cloning), and scales the contact buffer with ``num_envs``.
    """
    cfg.sim.dt = 1.0 / 120.0
    cfg.decimation = 8
    cfg.ctrl.disable_nullspace = True

    cfg_task = cfg.task
    kinematic_for = {"fixed_asset", "small_gear_cfg", "large_gear_cfg"}
    for asset_attr in ("fixed_asset", "held_asset", "small_gear_cfg", "large_gear_cfg"):
        asset_cfg = getattr(cfg_task, asset_attr, None)
        if isinstance(asset_cfg, ArticulationCfg):
            setattr(cfg_task, asset_attr, _to_rigid_object_cfg(asset_cfg, kinematic=asset_attr in kinematic_for))

    # Zero the bolt reset's Z noise: the bolt is a kinematic body (USD
    # PhysicsFixedJoint) so it sticks wherever the reset places it.
    noise = getattr(cfg_task, "fixed_asset_init_pos_noise", None)
    if noise is not None and len(noise) >= 3:
        cfg_task.fixed_asset_init_pos_noise = list(noise[:2]) + [0.0]

    # Bolt init z=0.0 matches the Newton cuboid table top.
    bolt = getattr(cfg_task, "fixed_asset", None)
    if bolt is not None and hasattr(bolt, "init_state") and hasattr(bolt.init_state, "pos"):
        cur_pos = bolt.init_state.pos
        new_init = bolt.init_state.replace(pos=(float(cur_pos[0]), float(cur_pos[1]), 0.0))
        cfg_task.fixed_asset = bolt.replace(init_state=new_init)

    # Arm armature stabilises OSC Lambda; must be set before super().__init__
    # so finalize bakes it into model.joint_armature.
    arm_actuators = cfg.robot.actuators
    arm_actuators["panda_arm1"].armature = 0.3
    arm_actuators["panda_arm2"].armature = 0.11
    arm_actuators["panda_hand"].armature = 0.15

    # kd=0: OSC Kd=2√Kp + armature already damp the 7th DOF; extra kd brakes
    # OSC tracking (kd=10 → +12 mm step error vs PhysX +48 mm).
    arm_actuators["panda_arm1"].damping = 0.0
    arm_actuators["panda_arm2"].damping = 0.0

    # Softer finger PD than PhysX default (7500, 173): with hydroelastic SDFs
    # grip force comes from kh × penetration, not the PD spring.
    arm_actuators["panda_hand"].stiffness = 1000.0
    arm_actuators["panda_hand"].damping = 10.0

    _monkey_patch_cloner_no_simplify()

    # Scale the contact buffer with num_envs so the gripper close doesn't
    # overflow when the user bumps the world count.
    if (
        getattr(cfg.sim.physics, "collision_cfg", None) is not None
        and getattr(cfg.sim.physics, "solver_cfg", None) is not None
        and hasattr(cfg.sim.physics.solver_cfg, "njmax")
    ):
        njmax = int(cfg.sim.physics.solver_cfg.njmax)
        num_worlds = int(cfg.scene.num_envs)
        cfg.sim.physics.collision_cfg.rigid_contact_max = njmax * num_worlds


def warm_up_kernels(env: FactoryEnv) -> None:
    """Pre-JIT Newton's per-step kernels by running two dummy steps.

    Factory's reset path runs a 30-iteration DLS IK loop where each iteration
    calls ``step_sim_no_action``. On the first iteration Newton JIT-compiles
    several kernels (``eval_fk``, the actuator-model kernels,
    ``eval_jacobian``/``eval_mass_matrix``) that each take 10-30 s; pre-paying
    that cost here keeps the 30 IK iterations on warm caches.
    """
    for _ in range(2):
        env.step_sim_no_action()


def _to_rigid_object_cfg(art_cfg: ArticulationCfg, kinematic: bool) -> RigidObjectCfg:
    """Adapt a joint-less Factory ArticulationCfg into a RigidObjectCfg.

    Wraps the asset as :class:`RigidObjectCfg` with
    ``kinematic_enabled=True`` for static targets so ``cfg.class_type(cfg)``
    constructs a :class:`RigidObject` at scene-setup time.
    """
    import isaaclab.sim as sim_utils  # noqa: PLC0415

    rigid_props = art_cfg.spawn.rigid_props
    if kinematic:
        rigid_props = (
            rigid_props.replace(kinematic_enabled=True)
            if rigid_props is not None
            else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        )
    spawn = art_cfg.spawn.replace(rigid_props=rigid_props)
    return RigidObjectCfg(
        prim_path=art_cfg.prim_path,
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=art_cfg.init_state.pos,
            rot=art_cfg.init_state.rot,
            lin_vel=art_cfg.init_state.lin_vel,
            ang_vel=art_cfg.init_state.ang_vel,
        ),
    )


def _monkey_patch_cloner_no_simplify() -> None:
    """Skip the cloner's convex-hull mesh approximation so per-shape SDFs survive."""
    from isaaclab_newton.cloner import newton_replicate as _nr

    _orig = _nr.newton_physics_replicate

    def _no_simplify(*args, **kwargs):
        kwargs.setdefault("simplify_meshes", False)
        return _orig(*args, **kwargs)

    _nr.newton_physics_replicate = _no_simplify
    import isaaclab_newton.cloner as _cloner_module  # noqa: PLC0415

    _cloner_module.newton_physics_replicate = _no_simplify


# ---------------------------------------------------------------------------
# Builder-time setup (MODEL_INIT callback body).
# ---------------------------------------------------------------------------


def _model_init_callback() -> None:
    """Body of the MODEL_INIT callback. Operates on the live builder."""
    from isaaclab_newton.physics import NewtonManager

    builder = NewtonManager._builder
    if builder is None:
        return

    _set_joint_target_mode(builder)
    _set_ctrl_source_joint_target(builder)
    _filter_base_table_contacts(builder)
    _tune_nut_bolt_contacts(builder)
    _build_collision_sdfs(builder)
    # NOTE: Per-DOF armature is set on the Franka via ``ImplicitActuatorCfg``
    # in :class:`FactoryEnvCfg.robot.actuators`, not via the builder.
    # Newton's finalize step seeds ``model.joint_armature`` for the robot's
    # DOFs from the actuator cfg and overwrites any per-DOF builder writes
    # we make at those indices. The actuator-cfg path is the only reliable
    # way to apply armature to a parsed-USD articulation today.


def _joint_label_indices(builder, name_substrs: list[str]) -> list[int]:
    """Return joint indices whose label contains any of ``name_substrs``."""
    return [i for i, label in enumerate(builder.joint_label) if any(s in label for s in name_substrs)]


def _joint_indices_to_dof_indices(builder, joint_idxs: list[int]) -> list[int]:
    """Translate joint indices into DOF indices via ``joint_qd_start``.

    Newton's per-DOF arrays (``joint_target_mode``, ``joint_target_ke``, …)
    are indexed by *DOF*, not joint. With free-floating joints in the
    scene (each contributes 6 qd entries but only 1 joint label),
    ``joint_index != dof_index`` past env 0. ``joint_qd_start[j]`` gives
    the qd-index where joint ``j``'s DOFs begin; ``joint_qd_start[j+1]``
    gives the end. We walk that range so a finger (1-DOF revolute) gives
    1 entry and a free joint (6-DOF) would give 6 — though we only call
    this for finite-DOF revolute joints.
    """
    qd_start = list(builder.joint_qd_start)
    total_dofs = qd_start[-1] if qd_start else 0
    out: list[int] = []
    for j in joint_idxs:
        dof_lo = int(qd_start[j])
        dof_hi = int(qd_start[j + 1]) if (j + 1) < len(qd_start) else int(total_dofs)
        out.extend(range(dof_lo, dof_hi))
    return out


def _arm_dof_indices(builder) -> list[int]:
    """DOF indices in ``builder.joint_target_mode`` for the arm joints."""
    js = _joint_label_indices(builder, _ARM_JOINT_NAMES)
    return _joint_indices_to_dof_indices(builder, js)


def _finger_dof_indices(builder) -> list[int]:
    """DOF indices in ``builder.joint_target_mode`` for the finger joints."""
    js = _joint_label_indices(builder, _FINGER_JOINT_NAMES)
    return _joint_indices_to_dof_indices(builder, js)


def _robot_body_indices(builder) -> list[int]:
    """Indices into ``builder.body_label`` that belong to the Franka."""
    return [i for i, label in enumerate(builder.body_label) if _ROBOT_BODY_PATH_SUBSTR in label]


def _set_joint_target_mode(builder) -> None:
    """Force every robot DOF (arm + fingers) into POSITION mode.

    With ``joint_target_ke = 0`` on the arm, the position target has no
    effect on the arm; but ``joint_target_kd > 0`` then gives mjwarp a
    per-joint ``-kd * qd`` damping term that kills the redundant-7th DOF
    mode the OSC would otherwise have to chase.
    """
    for dof_idx in _arm_dof_indices(builder) + _finger_dof_indices(builder):
        builder.joint_target_mode[dof_idx] = int(newton.JointTargetMode.POSITION)


def _set_ctrl_source_joint_target(builder) -> None:
    """Pin every robot DOF's ``mujoco:ctrl_source`` to ``JOINT_TARGET``.

    With ``mujoco:ctrl_source = JOINT_TARGET``, mjwarp reads the actuator's
    target from ``Control.joint_target_pos`` (which IsaacLab's
    ``set_joint_position_target_index`` writes) instead of the unused
    ``Control.mujoco.ctrl`` array. Required for the POSITION-mode arm
    to actually consume our position targets.
    """
    from newton.solvers import SolverMuJoCo

    custom = builder.custom_attributes.get("mujoco:ctrl_source")
    if custom is None:
        return
    n_acts = len(builder.joint_target_mode)
    if custom.values is None or len(custom.values) != n_acts:
        custom.values = [int(SolverMuJoCo.CtrlSource.JOINT_TARGET)] * n_acts
    else:
        target_value = int(SolverMuJoCo.CtrlSource.JOINT_TARGET)
        for dof_idx in _arm_dof_indices(builder) + _finger_dof_indices(builder):
            custom.values[dof_idx] = target_value


def _filter_base_table_contacts(builder) -> None:
    """Filter spurious robot-base ↔ table contacts.

    The Franka base (``panda_link0``, ``panda_link1``) sits on the table.
    Without explicit collision-filter pairs, mjwarp generates a continuous
    contact between the base collision mesh and the table top surface every
    step, even when there's no physical interpenetration. Those tiny contact
    forces propagate through the kinematic chain and show up as
    multi-millimeter TCP drift during OSC hold.
    """
    base_suffixes = ("/panda_link0", "/panda_link1")
    table_substr = "/Table"
    base_shape_idxs = [
        i
        for i, body_idx in enumerate(builder.shape_body)
        if 0 <= body_idx < len(builder.body_label) and builder.body_label[body_idx].endswith(base_suffixes)
    ]
    table_shape_idxs = [
        i
        for i, body_idx in enumerate(builder.shape_body)
        if 0 <= body_idx < len(builder.body_label) and table_substr in builder.body_label[body_idx]
    ]
    if not base_shape_idxs or not table_shape_idxs:
        return
    for base_i in base_shape_idxs:
        for table_i in table_shape_idxs:
            pair = (min(base_i, table_i), max(base_i, table_i))
            builder.shape_collision_filter_pairs.append(pair)
    logger.info("Filtered %d base<->table collision pairs.", len(base_shape_idxs) * len(table_shape_idxs))


def _tune_nut_bolt_contacts(builder) -> None:
    """Stiffen contact-material gains on every nut/bolt collision shape.

    Newton's parser defaults are too soft for the threading task: the gripper
    bounces off the nut. ``ke``/``kd`` lift normal stiffness/damping ~1-2 orders
    of magnitude; ``mu`` stays above ``MJ_MINMU`` to silence the NaN-risk warning.
    """
    if not all(hasattr(builder, attr) for attr in ("shape_label", "shape_material_mu", "shape_gap")):
        return
    for i in range(builder.shape_count):
        label = str(builder.shape_label[i]).lower()
        if "nut" in label:
            builder.shape_material_mu[i] = 0.2
            builder.shape_material_ke[i] = 1.0e4
            builder.shape_material_kd[i] = 100.0
            builder.shape_gap[i] = 0.0
        elif "bolt" in label:
            builder.shape_material_mu[i] = 0.5
            builder.shape_material_ke[i] = 1.0e4
            builder.shape_material_kd[i] = 100.0
            builder.shape_gap[i] = 0.0


# ---------------------------------------------------------------------------
# SDF / hydroelastic collision setup:
#   * fingers — 192-cube SDF + HYDROELASTIC, kh=1e11, condim=4.
#   * nut/bolt — 256-cube SDF + HYDROELASTIC, kh=1e11 (rigid thread features).
#   * other panda links — 64-cube SDF only, used for fast distance lookups.
# Builder-time only; the NewtonCfg (use_mujoco_contacts=False +
# sdf_hydroelastic_config) is what actually engages the hydroelastic forces.
# ---------------------------------------------------------------------------

# SDF resolutions (cube edge).
_SDF_RES_FINGER = 192
_SDF_RES_NUT_BOLT = 256
_SDF_RES_PANDA = 64
# Table is large + flat (1.2 × 0.6 × 0.04 m). 32³ → ~37 mm cells in xy,
# ~1.25 mm in z — plenty for a "rest on a flat surface" SDF without
# burning memory on a high-res grid.
_SDF_RES_TABLE = 32
_SDF_BAND_FINGER = (-0.01, 0.01)
_SDF_BAND_NUT_BOLT = (-0.005, 0.005)
_SDF_BAND_PANDA = (-0.01, 0.01)
# Wider outside band (2 cm) so the bolt + nut see the table SDF from
# slightly above, with a narrower inside band — nothing should ever
# be deep inside the table.
_SDF_BAND_TABLE = (-0.005, 0.02)

# Hydroelastic stiffness [Pa/m]. Rigid fingers + nut + bolt (1e11): at smaller
# values the finger SDF visibly punches through the nut SDF since the
# hydroelastic counter-force / penetration_volume can't push back the
# ~187 N from the finger PD close.
_KH_FINGER = 1e11
_KH_NUT_BOLT = 1e11

# Finger-only friction extras.
_FINGER_MU_TORSIONAL = 0.1
_FINGER_CONDIM = 4


def _build_collision_sdfs(builder) -> None:
    """Build SDFs on Factory's collision meshes for hydroelastic contacts.

    Categorises every collidable mesh shape by body label, builds an
    SDF at the resolution appropriate for that category, and (for
    fingers + nut + bolt) flips the ``HYDROELASTIC`` shape flag and
    writes ``shape_material_kh`` + ``shape_material_mu_torsional``.

    Idempotent per mesh: each :class:`newton.Mesh` whose ``sdf`` is
    already populated is skipped, so re-runs on a hot builder don't
    re-bake the SDF grids. Multiple shapes can share the same mesh,
    but mesh.build_sdf is only called once per unique mesh.
    """
    finger_names = ("panda_leftfinger", "panda_rightfinger")
    finger_body_idxs = {i for i, label in enumerate(builder.body_label) if any(n in label for n in finger_names)}
    nut_body_idxs = {i for i, label in enumerate(builder.body_label) if "HeldAsset/factory_nut_loose" in label}
    bolt_body_idxs = {i for i, label in enumerate(builder.body_label) if "FixedAsset/factory_bolt_loose" in label}
    table_body_idxs = {i for i, label in enumerate(builder.body_label) if "/Table" in label}
    panda_body_idxs = set(_robot_body_indices(builder)) - finger_body_idxs

    meshlike = (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH)
    counts = {
        "finger": 0,
        "nut": 0,
        "bolt": 0,
        "panda": 0,
        "table": 0,
        "skip_no_mesh": 0,
        "skip_already_built": 0,
    }

    condim_attr = builder.custom_attributes.get("mujoco:condim")
    if condim_attr is not None and condim_attr.values is None:
        condim_attr.values = {}

    for shape_idx, body_idx in enumerate(builder.shape_body):
        if int(builder.shape_type[shape_idx]) not in (int(t) for t in meshlike):
            continue
        if not (int(builder.shape_flags[shape_idx]) & int(newton.ShapeFlags.COLLIDE_SHAPES)):
            continue

        mesh = builder.shape_source[shape_idx]
        if mesh is None:
            counts["skip_no_mesh"] += 1
            continue

        if body_idx in finger_body_idxs:
            category = "finger"
            res, band = _SDF_RES_FINGER, _SDF_BAND_FINGER
        elif body_idx in nut_body_idxs:
            category = "nut"
            res, band = _SDF_RES_NUT_BOLT, _SDF_BAND_NUT_BOLT
        elif body_idx in bolt_body_idxs:
            category = "bolt"
            res, band = _SDF_RES_NUT_BOLT, _SDF_BAND_NUT_BOLT
        elif body_idx in panda_body_idxs:
            category = "panda"
            res, band = _SDF_RES_PANDA, _SDF_BAND_PANDA
        elif body_idx in table_body_idxs:
            # Voxel SDF for visualisation under "Show Collision". Not
            # flagged HYDROELASTIC — we want the bolt-on-table contact
            # to stay on Newton's default rigid pipeline so it doesn't
            # compete with the finger / nut / bolt hydroelastic mass
            # balance.
            category = "table"
            res, band = _SDF_RES_TABLE, _SDF_BAND_TABLE
        else:
            continue

        if mesh.sdf is None:
            # Bake non-unit shape_scale into the mesh vertices first —
            # ``mesh.build_sdf`` doesn't honour shape_scale, so a mesh
            # with shape_scale != 1 ends up with an SDF in the wrong
            # coordinate system. Resetting shape_scale to (1,1,1) afterwards
            # keeps shape vs. mesh in sync.
            shape_scale = builder.shape_scale[shape_idx]
            scale_arr = (float(shape_scale[0]), float(shape_scale[1]), float(shape_scale[2]))
            if not (
                abs(scale_arr[0] - 1.0) < 1e-6 and abs(scale_arr[1] - 1.0) < 1e-6 and abs(scale_arr[2] - 1.0) < 1e-6
            ):
                import numpy as _np  # noqa: PLC0415

                scaled_verts = mesh.vertices * _np.asarray(scale_arr, dtype=_np.float32)
                mesh = mesh.copy(vertices=scaled_verts, recompute_inertia=True)
                builder.shape_source[shape_idx] = mesh
                builder.shape_scale[shape_idx] = (1.0, 1.0, 1.0)
            mesh.build_sdf(max_resolution=res, narrow_band_range=band, margin=abs(band[1]))
            counts[category] += 1
        else:
            counts["skip_already_built"] += 1

        if category in ("finger", "nut", "bolt"):
            builder.shape_flags[shape_idx] |= int(newton.ShapeFlags.HYDROELASTIC)
            builder.shape_material_kh[shape_idx] = _KH_FINGER if category == "finger" else _KH_NUT_BOLT
        if category == "finger":
            builder.shape_material_mu_torsional[shape_idx] = _FINGER_MU_TORSIONAL
            if condim_attr is not None:
                condim_attr.values[shape_idx] = _FINGER_CONDIM

    logger.info(
        "Built SDFs: finger=%d nut=%d bolt=%d panda=%d table=%d (skipped: no_mesh=%d, already_built=%d).",
        counts["finger"],
        counts["nut"],
        counts["bolt"],
        counts["panda"],
        counts["table"],
        counts["skip_no_mesh"],
        counts["skip_already_built"],
    )
