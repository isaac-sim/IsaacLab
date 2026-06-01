# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory Newton-only setup: cfg overrides, model-init callback, kernel warm-up."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import newton

from isaaclab.assets import ArticulationCfg, RigidObjectCfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .factory_env import FactoryEnv
    from .factory_env_cfg import FactoryEnvCfg

_ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
_FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
_ROBOT_BODY_PATH_SUBSTR = "/Robot/"

# Latched in :func:`apply_cfg_overrides`; read by the MODEL_INIT callback.
_use_hydroelastic: bool = True


def register_model_init_callback() -> None:
    """Wire a Newton MODEL_INIT callback that mutates the builder."""
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.physics import PhysicsEvent

    NewtonManager.register_callback(
        lambda _ev: _model_init_callback(), PhysicsEvent.MODEL_INIT, name="factory_newton_setup"
    )


def apply_cfg_overrides(cfg: FactoryEnvCfg) -> None:
    """Apply Newton-only cfg overrides in place, before ``super().__init__``."""
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

    # Arm armature stabilises OSC Lambda — must be set pre-super().__init__.
    arm_actuators = cfg.robot.actuators
    arm_actuators["panda_arm1"].armature = 0.3
    arm_actuators["panda_arm2"].armature = 0.11
    arm_actuators["panda_hand"].armature = 0.15

    # OSC kd already damps via 2√Kp + armature; extra joint kd brakes tracking.
    arm_actuators["panda_arm1"].damping = 0.0
    arm_actuators["panda_arm2"].damping = 0.0

    global _use_hydroelastic
    _use_hydroelastic = getattr(cfg.sim.physics.collision_cfg, "sdf_hydroelastic_config", None) is not None

    if _use_hydroelastic:
        arm_actuators["panda_hand"].stiffness = 1000.0
        arm_actuators["panda_hand"].damping = 10.0
    else:
        kp = float(os.environ.get("FACTORY_SDF_FINGER_KP", "8000"))
        kd = float(os.environ.get("FACTORY_SDF_FINGER_KD", "160"))
        arm_actuators["panda_hand"].stiffness = kp
        arm_actuators["panda_hand"].damping = kd

    solver_cfg = getattr(cfg.sim.physics, "solver_cfg", None)
    if solver_cfg is not None:
        iters_override = int(os.environ.get("FACTORY_SOLVER_ITERATIONS", "0"))
        imp_override = float(os.environ.get("FACTORY_SOLVER_IMPRATIO", "0"))
        if iters_override > 0 and hasattr(solver_cfg, "iterations"):
            solver_cfg.iterations = iters_override
        if imp_override > 0 and hasattr(solver_cfg, "impratio"):
            solver_cfg.impratio = imp_override

    substeps_override = int(os.environ.get("FACTORY_NUM_SUBSTEPS", "0"))
    if substeps_override > 0 and hasattr(cfg.sim.physics, "num_substeps"):
        cfg.sim.physics.num_substeps = substeps_override

    _monkey_patch_cloner_no_simplify()

    # Scale contact buffer with num_envs; SDF-only emits ~3x hydroelastic.
    if (
        getattr(cfg.sim.physics, "collision_cfg", None) is not None
        and getattr(cfg.sim.physics, "solver_cfg", None) is not None
        and hasattr(cfg.sim.physics.solver_cfg, "njmax")
    ):
        njmax_mult = float(os.environ.get("FACTORY_NJMAX_MULT", "1.0"))
        njmax = int(cfg.sim.physics.solver_cfg.njmax)
        num_worlds = int(cfg.scene.num_envs)
        cfg.sim.physics.collision_cfg.rigid_contact_max = int(njmax * num_worlds * njmax_mult)


def warm_up_kernels(env: FactoryEnv) -> None:
    """Pre-JIT Newton's per-step kernels so the reset IK loop runs on warm caches."""
    for _ in range(2):
        env.step_sim_no_action()


def _to_rigid_object_cfg(art_cfg: ArticulationCfg, kinematic: bool) -> RigidObjectCfg:
    """Adapt a joint-less Factory ArticulationCfg into a RigidObjectCfg."""
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
    # Armature is applied via ImplicitActuatorCfg, not the builder — finalize
    # overwrites builder writes at robot DOF indices with the actuator value.


def _joint_label_indices(builder, name_substrs: list[str]) -> list[int]:
    """Return joint indices whose label contains any of ``name_substrs``."""
    return [i for i, label in enumerate(builder.joint_label) if any(s in label for s in name_substrs)]


def _joint_indices_to_dof_indices(builder, joint_idxs: list[int]) -> list[int]:
    """Translate joint indices into DOF indices via ``joint_qd_start``.

    Free-floating joints contribute 6 qd entries per joint, so
    ``joint_index != dof_index`` past env 0; walk ``joint_qd_start`` to
    cover every DOF the joint owns.
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
    """Force every robot DOF (arm + fingers) into POSITION mode."""
    for dof_idx in _arm_dof_indices(builder) + _finger_dof_indices(builder):
        builder.joint_target_mode[dof_idx] = int(newton.JointTargetMode.POSITION)


def _set_ctrl_source_joint_target(builder) -> None:
    """Pin every robot DOF's ``mujoco:ctrl_source`` to ``JOINT_TARGET``.

    Required so mjwarp reads targets from ``Control.joint_target_pos``
    (set by ``set_joint_position_target_index``) rather than the unused
    ``Control.mujoco.ctrl`` array.
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

    Without this filter mjwarp generates continuous tiny contacts between
    base links and the table top; they propagate through the chain and
    show up as multi-millimeter TCP drift during OSC hold.
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
    """Tune contact-material gains on every nut/bolt collision shape.

    Hydroelastic mode keeps the penalty spring soft so it doesn't
    double-count the hydroelastic normal force; SDF-only inherits the
    Newton nut/bolt example values (ke=1e7, kd=1e4, gap=5 mm).
    """
    if not all(hasattr(builder, attr) for attr in ("shape_label", "shape_material_mu", "shape_gap")):
        return

    if _use_hydroelastic:
        ke, kd, gap = 1.0e4, 100.0, 0.0
    else:
        ke = float(os.environ.get("FACTORY_SDF_NUT_BOLT_KE", "1.0e7"))
        kd = float(os.environ.get("FACTORY_SDF_NUT_BOLT_KD", "1.0e4"))
        gap = float(os.environ.get("FACTORY_SDF_NUT_BOLT_GAP", "0.005"))

    for i in range(builder.shape_count):
        label = str(builder.shape_label[i]).lower()
        if "nut" in label:
            builder.shape_material_mu[i] = 0.2
            builder.shape_material_ke[i] = ke
            builder.shape_material_kd[i] = kd
            builder.shape_gap[i] = gap
        elif "bolt" in label:
            builder.shape_material_mu[i] = 0.5
            builder.shape_material_ke[i] = ke
            builder.shape_material_kd[i] = kd
            builder.shape_gap[i] = gap


# ---------------------------------------------------------------------------
# SDF collision setup. Both modes build the same SDF grids; only the
# per-shape material flags differ:
#   * hydroelastic: HYDROELASTIC flag + ``kh`` on finger/nut/bolt shapes.
#   * sdf-only:     ``ke`` / ``kd`` penalty springs on finger shapes.
# Finger friction extras (torsional + condim=4) apply in both modes.
# ---------------------------------------------------------------------------

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
# Wider outside band so bolt + nut see the table SDF from slightly above.
_SDF_BAND_TABLE = (-0.005, 0.02)

# Hydroelastic stiffness [Pa/m]. Smaller values let the finger SDF punch
# through the nut SDF under the ~187 N PD close force.
_KH_FINGER = 1e11
_KH_NUT_BOLT = 1e11

# SDF-only finger penalty spring. The Newton nut/bolt example ships
# ke=1e7/kd=1e4, but its scene has no gripper; finger pads under the
# saturated 40 N close force need ~30× stiffer contact. Sweep picked
# (3e8, 3e5) on Factory_develop_baseline.pth (success 0.625 → 0.891);
# higher values oscillate.
_KE_FINGER = 3.0e8
_KD_FINGER = 3.0e5

_FINGER_MU_TORSIONAL = 0.1
_FINGER_CONDIM = 4


def _build_collision_sdfs(builder) -> None:
    """Bake per-shape SDFs and tag finger/nut/bolt for the active contact model.

    Idempotent: meshes with a populated ``sdf`` are skipped, so re-runs
    on a hot builder don't rebake.
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
            # Voxel SDF for "Show Collision"; not HYDROELASTIC — keeps
            # bolt-on-table contact off the hydroelastic mass balance.
            category = "table"
            res, band = _SDF_RES_TABLE, _SDF_BAND_TABLE
        else:
            continue

        if mesh.sdf is None:
            # build_sdf doesn't honour shape_scale, so bake it into vertices.
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

        if _use_hydroelastic and category in ("finger", "nut", "bolt"):
            builder.shape_flags[shape_idx] |= int(newton.ShapeFlags.HYDROELASTIC)
            builder.shape_material_kh[shape_idx] = _KH_FINGER if category == "finger" else _KH_NUT_BOLT
        elif not _use_hydroelastic and category == "finger":
            builder.shape_material_ke[shape_idx] = float(
                os.environ.get("FACTORY_SDF_FINGER_CONTACT_KE", str(_KE_FINGER))
            )
            builder.shape_material_kd[shape_idx] = float(
                os.environ.get("FACTORY_SDF_FINGER_CONTACT_KD", str(_KD_FINGER))
            )
            finger_gap_env = os.environ.get("FACTORY_SDF_FINGER_GAP")
            if finger_gap_env is not None:
                builder.shape_gap[shape_idx] = float(finger_gap_env)
        if category == "finger":
            builder.shape_material_mu_torsional[shape_idx] = _FINGER_MU_TORSIONAL
            if condim_attr is not None:
                condim_attr.values[shape_idx] = _FINGER_CONDIM

    logger.info(
        "Built SDFs (%s): finger=%d nut=%d bolt=%d panda=%d table=%d (skipped: no_mesh=%d, already_built=%d).",
        "hydroelastic" if _use_hydroelastic else "sdf-only",
        counts["finger"],
        counts["nut"],
        counts["bolt"],
        counts["panda"],
        counts["table"],
        counts["skip_no_mesh"],
        counts["skip_already_built"],
    )
