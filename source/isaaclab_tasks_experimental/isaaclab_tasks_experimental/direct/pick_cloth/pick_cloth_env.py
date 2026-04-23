# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-Cloth environment: Franka robot interacts with cloth using a coupled solver."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import warp as wp

from isaaclab_contrib.deformable import register_hooks as _register_deformable_hooks

_register_deformable_hooks()

from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets.deformable_object import DeformableObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.shapes import SphereCfg, spawn_sphere
from isaaclab.sim.utils.stage import get_current_stage

from .pick_cloth_env_cfg import PickClothEnvCfg

logger = logging.getLogger(__name__)


class PickClothEnv(DirectRLEnv):
    cfg: PickClothEnvCfg

    def __init__(self, cfg: PickClothEnvCfg, render_mode: str | None = None, **kwargs):
        self._has_robot = cfg.robot_cfg is not None

        # Without a robot the coupled solver (rigid + VBD) is unnecessary and will
        # fail because there are no rigid bodies.  Swap to VBD-only automatically.
        if not self._has_robot:
            from isaaclab_contrib.deformable.newton_manager_cfg import CoupledSolverCfg

            physics_cfg = cfg.sim.physics
            if hasattr(physics_cfg, "solver_cfg") and isinstance(physics_cfg.solver_cfg, CoupledSolverCfg):
                physics_cfg.solver_cfg = physics_cfg.solver_cfg.vbd_cfg

        # For velocity control, override actuator gains before the robot is spawned:
        # zero stiffness (no position tracking), high damping (velocity-tracking gain).
        # Featherstone torque: tau = ke*(pos_target - q) + kd*(vel_target - qd)
        # With ke=0: tau = kd*(vel_target - qd)  -- proportional velocity control.
        if self._has_robot and cfg.control_mode == "velocity":
            for actuator in cfg.robot_cfg.actuators.values():
                actuator.stiffness = 0.0
                actuator.damping = 200.0

        super().__init__(cfg, render_mode, **kwargs)

        if self._has_robot:
            self._arm_joint_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names)
            self._default_joint_pos = wp.to_torch(self.robot.data.default_joint_pos).clone()

            self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
            self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

            # Find EE body index for reward computation
            ee_body_idx, _ = self.robot.find_bodies("panda_hand")
            self._ee_body_idx = int(ee_body_idx[0])

        # Keyboard controls (Newton viewer)
        self._request_reset = False
        self._gripper_closed = False
        self._reset_key_registered = False
        self._newton_viewer_gl = None

        # Finger joint indices and limits
        if self._has_robot:
            self._finger_joint_idx, _ = self.robot.find_joints(["panda_finger_joint1", "panda_finger_joint2"])

        # ------------------------------------------------------------------
        # Optional interactive IK -- Newton backend only
        # ------------------------------------------------------------------
        self._ik_available = False
        if self._has_robot and cfg.interactive_ik:
            self._setup_interactive_ik()

        logger.info(
            "PickClothEnv: has_robot=%s, control_mode=%s, action_scale=%s, interactive_ik=%s",
            self._has_robot,
            self.cfg.control_mode,
            cfg.action_scale,
            self._ik_available,
        )

    _SPHERE_PRIM_PATH = "/World/ik_target"

    def _setup_interactive_ik(self):
        """Initialize Newton IK solver and the draggable target sphere."""
        try:
            import newton
            import newton.ik as ik
            from isaaclab_newton.physics import NewtonManager

            newton_model = NewtonManager._model
            if newton_model is None:
                logger.info("[PickClothEnv] Newton model not available; IK disabled.")
                return

            ee_body_idx, _ = self.robot.find_bodies("panda_hand")
            self._ee_ik_index = int(ee_body_idx[0])

            # Write robot default joint positions into model before FK
            default_jpos = wp.to_torch(self.robot.data.default_joint_pos)[0]
            joint_q_torch = wp.to_torch(newton_model.joint_q)
            n_robot = min(default_jpos.shape[0], joint_q_torch.shape[0])
            joint_q_torch[:n_robot] = default_jpos[:n_robot]

            # Compute initial EE transform via FK
            ik_state = newton_model.state()
            newton.eval_fk(newton_model, newton_model.joint_q, newton_model.joint_qd, ik_state)
            body_q_np = ik_state.body_q.numpy()
            self._ee_tf = wp.transform(*body_q_np[self._ee_ik_index])
            ee_pos = wp.transform_get_translation(self._ee_tf)

            # IK objectives
            ee_rot = wp.transform_get_rotation(self._ee_tf)
            self._pos_obj = ik.IKObjectivePosition(
                link_index=self._ee_ik_index,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([ee_pos], dtype=wp.vec3),
            )
            self._rot_obj = ik.IKObjectiveRotation(
                link_index=self._ee_ik_index,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array([wp.vec4(ee_rot[0], ee_rot[1], ee_rot[2], ee_rot[3])], dtype=wp.vec4),
            )
            self._joint_limit_obj = ik.IKObjectiveJointLimit(
                joint_limit_lower=newton_model.joint_limit_lower,
                joint_limit_upper=newton_model.joint_limit_upper,
                weight=0.0,
            )

            self._ik_joint_q = wp.array(newton_model.joint_q, shape=(1, newton_model.joint_coord_count))
            self._ik_solver = ik.IKSolver(
                model=newton_model,
                n_problems=1,
                objectives=[self._pos_obj, self._rot_obj, self._joint_limit_obj],
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )
            self._newton_model = newton_model
            self._ik_available = True
            self._newton_viewer_gl = None  # lazily resolved in _apply_action
            logger.info("[PickClothEnv] Newton IK initialized (EE index=%d)", self._ee_ik_index)

            # Spawn IK target sphere and teleport to EE position
            self._stage = get_current_stage()
            spawn_sphere(
                self._SPHERE_PRIM_PATH,
                SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                translation=(float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])),
            )
            self._sphere_prim = self._stage.GetPrimAtPath(self._SPHERE_PRIM_PATH)
        except Exception as exc:
            logger.info("[PickClothEnv] IK not available: %s", exc)

    def _apply_ik_action(self):
        """Read gizmo target, solve IK, and set joint position targets."""
        # Lazily find Newton viewer for gizmo
        if self._newton_viewer_gl is None:
            try:
                from isaaclab_visualizers.newton import NewtonVisualizer

                for v in self.sim.visualizers:
                    if isinstance(v, NewtonVisualizer) and v._viewer is not None:
                        self._newton_viewer_gl = v._viewer
                        break
            except Exception:
                pass

            if self._newton_viewer_gl is not None:
                self._register_reset_key()

                # Patch begin_frame to register the draggable gizmo each render step
                _orig_bf = self._newton_viewer_gl.begin_frame
                _tf = self._ee_tf
                _viewer = self._newton_viewer_gl

                def _begin_frame_with_gizmo(time, _orig=_orig_bf, _v=_viewer, _t=_tf):
                    _orig(time)
                    _v.log_gizmo("ik_target", _t)

                self._newton_viewer_gl.begin_frame = _begin_frame_with_gizmo
                logger.info("[PickClothEnv] Newton viewer gizmo registered")
            else:
                logger.warning("[PickClothEnv] NewtonViewerGL not found in sim.visualizers")

        if self._newton_viewer_gl is not None:
            # Render a red sphere at the IK target position
            device = self._newton_viewer_gl.device
            target_pos_vec = wp.transform_get_translation(self._ee_tf)
            self._newton_viewer_gl.log_points(
                "ik_target_sphere",
                points=wp.array([target_pos_vec], dtype=wp.vec3, device=device),
                radii=wp.array([0.05], dtype=wp.float32, device=device),
                colors=wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            )

        # Sync USD sphere prim to the (possibly gizmo-updated) target position
        target_pos = wp.transform_get_translation(self._ee_tf)
        xform = UsdGeom.Xformable(self._sphere_prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])))
                break

        # Warm-start IK from current joint state
        current_q = wp.to_torch(self.robot.data.joint_pos)[0]
        ik_q_torch = wp.to_torch(self._ik_joint_q)
        n_robot = current_q.shape[0]
        ik_q_torch[0, :n_robot] = current_q[:n_robot]

        self._pos_obj.set_target_position(0, wp.transform_get_translation(self._ee_tf))
        ee_rot = wp.transform_get_rotation(self._ee_tf)
        self._rot_obj.set_target_rotation(0, ee_rot)
        self._ik_solver.step(self._ik_joint_q, self._ik_joint_q, iterations=24)

        solved_arm_q = wp.to_torch(self._ik_joint_q)[0, self._arm_joint_idx]
        self.robot.set_joint_position_target_index(target=solved_arm_q.unsqueeze(0), joint_ids=self._arm_joint_idx)

        # Finger control: closed=0.0, open=0.04 (joint limits)
        self._apply_finger_targets()

    def _apply_finger_targets(self):
        """Set finger joint targets based on gripper state (G key toggle)."""
        finger_pos = 0.0 if self._gripper_closed else 0.04
        finger_target = torch.full(
            (self.num_envs, len(self._finger_joint_idx)),
            finger_pos,
            dtype=torch.float32,
            device=self.device,
        )
        self.robot.set_joint_position_target_index(target=finger_target, joint_ids=self._finger_joint_idx)

    def _try_find_viewer_for_reset_key(self):
        """Lazily find Newton viewer and register R-key reset (non-IK path)."""
        if self._reset_key_registered:
            return
        try:
            from isaaclab_visualizers.newton import NewtonVisualizer

            for v in self.sim.visualizers:
                if isinstance(v, NewtonVisualizer) and v._viewer is not None:
                    self._newton_viewer_gl = v._viewer
                    self._register_reset_key()
                    return
        except Exception:
            pass

    def _register_reset_key(self):
        """Register R key to trigger environment reset in Newton viewer."""
        if self._reset_key_registered or self._newton_viewer_gl is None:
            return
        import pyglet.window.key as key

        def _on_key(symbol, modifiers, _self=self):
            if symbol == key.R:
                _self._request_reset = True
                logger.info("[PickClothEnv] Reset requested via R key")
            elif symbol == key.G:
                _self._gripper_closed = not _self._gripper_closed
                logger.info("[PickClothEnv] Gripper %s via G key", "closed" if _self._gripper_closed else "open")

        self._newton_viewer_gl.renderer.register_key_press(_on_key)
        self._reset_key_registered = True
        logger.info("[PickClothEnv] R key (reset) and G key (gripper toggle) registered")

    def _setup_scene(self):
        # Robot (optional)
        if self._has_robot:
            self.robot = Articulation(self.cfg.robot_cfg)

        # Cloth asset (triangle surface mesh)
        self.cloth = DeformableObject(self.cfg.cloth)

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self._has_robot:
            self.scene.articulations["robot"] = self.robot

        if self._has_robot and self.cfg.disable_robot_ground_collision:
            self._register_ground_collision_disable()

    def _register_ground_collision_disable(self):
        """Register a PHYSICS_READY callback to disable ground-robot collision.

        Sets the ground plane's ``shape_collision_group`` to 0 in Newton, which
        disables rigid-body collisions while keeping soft (particle) contacts active.
        """
        from isaaclab_newton.physics import NewtonManager

        from isaaclab.physics import PhysicsEvent

        def _disable(payload=None):
            model = NewtonManager._model
            if model is None:
                return
            labels = list(model.shape_label)
            groups = model.shape_collision_group.numpy()
            for i, label in enumerate(labels):
                if "ground" in label.lower() or "defaultgroundplane" in label.lower():
                    groups[i] = 0
            model.shape_collision_group.assign(wp.array(groups, dtype=int, device=model.shape_collision_group.device))
            logger.info(
                "Disabled ground-robot collision for shapes: %s",
                [labels[i] for i, g in enumerate(groups) if g == 0],
            )

        NewtonManager.register_callback(_disable, PhysicsEvent.PHYSICS_READY)

    # --- RL interface ---

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Lazily register R-key reset when Newton viewer is available (non-IK path)
        if not self._reset_key_registered and not self._ik_available:
            self._try_find_viewer_for_reset_key()

        if not self._has_robot:
            return
        if self._ik_available:
            self._apply_ik_action()
            return
        if self.cfg.control_mode == "velocity":
            # Velocity control: actions are target joint velocities [rad/s]
            vel_targets = self.actions * self.cfg.action_scale
            self.robot.set_joint_velocity_target_index(target=vel_targets, joint_ids=self._arm_joint_idx)
        else:
            # Position control: actions are offsets from default pose [rad]
            pos_targets = self._default_joint_pos[:, self._arm_joint_idx] + self.actions * self.cfg.action_scale
            self.robot.set_joint_position_target_index(target=pos_targets, joint_ids=self._arm_joint_idx)
        self._apply_finger_targets()

    def _get_observations(self) -> dict:
        self.cloth.update(self.step_dt)

        # Cloth centroid: mean of all nodal positions
        nodal_pos = wp.to_torch(self.cloth.data.nodal_pos_w)  # (num_envs, num_particles, 3)
        self._cloth_centroid = nodal_pos.mean(dim=1)  # (num_envs, 3)

        if self._has_robot:
            obs = torch.cat(
                (
                    self.joint_pos[:, self._arm_joint_idx],  # (num_envs, 7)
                    self.joint_vel[:, self._arm_joint_idx],  # (num_envs, 7)
                    self._cloth_centroid,  # (num_envs, 3)
                ),
                dim=-1,
            )
        else:
            obs = self._cloth_centroid  # (num_envs, 3)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # Cloth height reward -- encourage lifting
        cloth_height = self._cloth_centroid[:, 2]  # z-component
        rew_cloth_height = self.cfg.rew_scale_cloth_height * cloth_height

        if self._has_robot:
            # EE-to-cloth distance penalty -- encourage reaching toward cloth
            ee_pos = wp.to_torch(self.robot.data.body_pos_w)[:, self._ee_body_idx]  # (num_envs, 3)
            ee_cloth_dist = torch.norm(ee_pos - self._cloth_centroid, dim=-1)
            rew_ee_cloth_dist = self.cfg.rew_scale_ee_cloth_dist * ee_cloth_dist

            # Joint velocity penalty
            rew_joint_vel = self.cfg.rew_scale_joint_vel * torch.sum(
                torch.abs(self.joint_vel[:, self._arm_joint_idx]), dim=-1
            )
            return rew_cloth_height + rew_ee_cloth_dist + rew_joint_vel

        return rew_cloth_height

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._has_robot:
            self.joint_pos = wp.to_torch(self.robot.data.joint_pos)
            self.joint_vel = wp.to_torch(self.robot.data.joint_vel)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # R key in Newton viewer triggers reset for all envs
        if self._request_reset:
            time_out = torch.ones_like(time_out)
            self._request_reset = False

        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        if self._has_robot:
            # Reset robot joint state to defaults
            joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
            joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids].clone()

            # Root pose -- add env origins for world frame
            default_root_pose = wp.to_torch(self.robot.data.default_root_pose)[env_ids].clone()
            default_root_pose[:, :3] += self.scene.env_origins[env_ids]
            default_root_vel = wp.to_torch(self.robot.data.default_root_vel)[env_ids].clone()

            # Update cached views
            self.joint_pos[env_ids] = joint_pos
            self.joint_vel[env_ids] = joint_vel

            # Write robot state to simulation
            self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
            self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
            self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Run FK so body_q matches the reset joint_q before the next simulation step.
        self.sim.forward()

        # Sync rigid-body state from state_0 to state_1 for reset envs only.
        # VBD (with integrate_with_external_rigid_solver=True) reads body_q from state_out,
        # which is state_1 in the first substep. Without this sync, state_1 retains the
        # pre-reset robot pose and cloth particles see stale body contacts.
        from isaaclab_newton.physics import NewtonManager as SimulationManager

        s0 = SimulationManager.get_state_0()
        s1 = SimulationManager.get_state_1()
        model = SimulationManager.get_model()

        if s0.body_q is not None and s1.body_q is not None:
            bodies_per_env = model.body_count // model.world_count
            s0_bq = wp.to_torch(s0.body_q).view(model.world_count, bodies_per_env, -1)
            s1_bq = wp.to_torch(s1.body_q).view(model.world_count, bodies_per_env, -1)
            s1_bq[env_ids] = s0_bq[env_ids]

        if s0.body_qd is not None and s1.body_qd is not None:
            bodies_per_env = model.body_count // model.world_count
            s0_bqd = wp.to_torch(s0.body_qd).view(model.world_count, bodies_per_env, -1)
            s1_bqd = wp.to_torch(s1.body_qd).view(model.world_count, bodies_per_env, -1)
            s1_bqd[env_ids] = s0_bqd[env_ids]

        if s0.joint_q is not None and s1.joint_q is not None:
            dofs_per_env = model.joint_dof_count // model.world_count
            s0_jq = wp.to_torch(s0.joint_q).view(model.world_count, dofs_per_env)
            s1_jq = wp.to_torch(s1.joint_q).view(model.world_count, dofs_per_env)
            s1_jq[env_ids] = s0_jq[env_ids]

        if s0.joint_qd is not None and s1.joint_qd is not None:
            dofs_per_env = model.joint_dof_count // model.world_count
            s0_jqd = wp.to_torch(s0.joint_qd).view(model.world_count, dofs_per_env)
            s1_jqd = wp.to_torch(s1.joint_qd).view(model.world_count, dofs_per_env)
            s1_jqd[env_ids] = s0_jqd[env_ids]

        # Reset cloth: restore default nodal state (positions + zero velocities).
        # reset() is a no-op (matches PhysX convention); state is restored via write.
        env_ids_list = env_ids.cpu().tolist() if hasattr(env_ids, "cpu") else list(env_ids)
        default_state = wp.to_torch(self.cloth.data.default_nodal_state_w)
        self.cloth.write_nodal_state_to_sim_index(default_state, env_ids=env_ids_list)
        self.cloth.reset(env_ids=env_ids_list)
