# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ik_7d inverse-kinematics controller for Isaac Lab.

ik_7d is AgiBot's closed-form-seeded SQP solver for redundant 7-DoF arms. Unlike
a generic differential-IK solver it exposes the redundancy as a *named
coordinate* -- ``arm_plane_angle``, the elbow swivel about the shoulder-wrist
axis -- so the null space can be commanded rather than merely tolerated.

Reference:
    ik_7d 0.3.5 (AgiBot). The wheel is shipped alongside this extension.

Notes:
    ik_7d is a CPU/numpy solver and holds mutable model state, so there is one
    instance per environment and the solves are serial. At ~0.20 ms median /
    0.63 ms max per arm this is comfortable for a handful of environments and a
    teleoperation loop; it is not a training-scale action term.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from ik_7d import IK7D, Group, Pose7D

if TYPE_CHECKING:
    from .ik_7d_controller_cfg import Ik7dControllerCfg


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``.

    ``IK7D.armPlaneAngle`` reports in this range but commands are not wrapped.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class JointMap:
    """Bidirectional index map between Isaac Lab joint order and ik_7d's model order.

    Isaac Lab orders ``Articulation.data.joint_names`` by PhysX traversal of the
    USD, which matches neither the URDF's declaration order nor ik_7d's. The
    permutation is built at runtime from names -- a hard-coded index table
    surfaces as the robot moving the wrong joints -- and a partial map is refused
    rather than returned.
    """

    def __init__(self, lab_joint_names: list[str], model_joint_names: list[str]):
        self.lab_names = list(lab_joint_names)
        self.model_names = list(model_joint_names)

        lab_index: dict[str, int] = {}
        for i, name in enumerate(self.lab_names):
            if name in lab_index:
                raise ValueError(f"duplicate joint name in the Lab articulation: {name!r}")
            lab_index[name] = i

        missing = [n for n in self.model_names if n not in lab_index]
        if missing:
            raise ValueError(
                f"{len(missing)} ik_7d model joint(s) absent from the Lab articulation: "
                f"{missing}. The USD was probably converted from a different URDF."
            )

        # model_to_lab[k] is the Lab index of the k-th ik_7d model joint.
        self.model_to_lab = np.asarray([lab_index[n] for n in self.model_names], dtype=np.int64)
        self.unmapped_lab = [n for n in self.lab_names if n not in set(self.model_names)]

        if len(set(self.model_to_lab.tolist())) != len(self.model_names):
            raise ValueError("mapping is not injective")

    def __len__(self) -> int:
        return len(self.model_names)

    def to_model(self, lab_q: np.ndarray) -> np.ndarray:
        """Gather a full Lab joint vector into ik_7d's model order."""
        return np.asarray(lab_q, dtype=np.float64)[self.model_to_lab]

    def lab_indices_for(self, model_indices: np.ndarray) -> np.ndarray:
        """Lab indices for a subset of model joints, e.g. one arm's group.

        This is what the action term hands to ``set_joint_position_target_index``.
        """
        return self.model_to_lab[np.asarray(model_indices, dtype=np.int64)]

    def describe(self) -> str:
        out = f"{len(self.model_names)} ik_7d joints mapped into {len(self.lab_names)} Lab joints"
        if self.unmapped_lab:
            out += f"; not modelled by ik_7d ({len(self.unmapped_lab)}): " + ", ".join(self.unmapped_lab)
        return out


class Ik7dController:
    """One ik_7d solver bound to one environment's articulation.

    The controller owns the Lab-to-model index permutation, the per-arm seeds and
    the failure-hold state. It is deliberately numpy-in / numpy-out: the action
    term handles all tensor and frame work, so this class can be unit-tested with
    no Isaac Sim running.
    """

    ARM_GROUPS: dict[str, Group] = {"left": Group.Left_Arm, "right": Group.Right_Arm}
    """Arm name to ik_7d group. Keys are what :attr:`Ik7dControllerCfg.arms` accepts."""

    def __init__(self, cfg: Ik7dControllerCfg, lab_joint_names: list[str]):
        """Build the solver and resolve the joint mapping.

        Args:
            cfg: Controller configuration.
            lab_joint_names: ``Articulation.data.joint_names``, in Lab order.

        Raises:
            ValueError: If an arm name is unknown, or if any ik_7d model joint is
                absent from the articulation.
        """
        unknown = [a for a in cfg.arms if a not in self.ARM_GROUPS]
        if unknown:
            raise ValueError(f"unknown arm(s) {unknown}; expected a subset of {list(self.ARM_GROUPS)}")

        self.cfg = cfg
        self._ik = IK7D(cfg.robot_name, cfg.urdf_path)

        model = self._ik.getModelInfo()
        self.model_names = list(model.joint_names)
        self.jmap = JointMap(lab_joint_names, self.model_names)

        # ik_7d ships a home pose that violates its own IK limits (G2_t2_crs both
        # arms' joint3 is home 0.0 against a band excluding zero). Project once; an
        # unprojected seed snaps to the limit on the first solve and reads as a
        # phantom discontinuity in every trajectory that starts here.
        self._lb = np.asarray(model.lb_ik, dtype=np.float64)
        self._ub = np.asarray(model.ub_ik, dtype=np.float64)
        self.home_model = np.clip(np.asarray(model.home_joints, dtype=np.float64), self._lb, self._ub)
        self._ik.setModelState(self.home_model)

        self.model_indices: dict[str, np.ndarray] = {}
        self.lab_indices: dict[str, np.ndarray] = {}
        self.base_frame: dict[str, str] = {}
        self.ee_frame: dict[str, str] = {}
        self.home_apa: dict[str, float] = {}
        self.free_sign: dict[str, int] = {}
        self._seed: dict[str, np.ndarray] = {}
        self._last_good: dict[str, np.ndarray] = {}

        for arm in cfg.arms:
            group = self.ARM_GROUPS[arm]
            indices = np.asarray(self._ik.getGroupInfo(group).joint_indices, dtype=np.int64)
            self.model_indices[arm] = indices
            self.lab_indices[arm] = self.jmap.lab_indices_for(indices)

            home_arm = self.home_model[indices].copy()
            fk = self._ik.fk7d(group, home_arm)
            self.base_frame[arm] = fk.base_frame
            self.ee_frame[arm] = fk.frame
            self.home_apa[arm] = float(self._ik.armPlaneAngle(group, home_arm))
            self._seed[arm] = home_arm
            self._last_good[arm] = home_arm.copy()
            self.free_sign[arm] = self.probe_free_sign(arm) if cfg.apa_mode == "free_delta" else 1

        # Flat Lab indices of every joint this controller writes, in action order.
        self.controlled_lab_indices: list[int] = [int(i) for arm in cfg.arms for i in self.lab_indices[arm]]

        self.solves = 0
        self.qp_failures = 0
        self._last_warned = 0.0

    # ==================== Properties ====================

    @property
    def num_arms(self) -> int:
        """Number of arms this controller drives."""
        return len(self.cfg.arms)

    @property
    def num_controlled_joints(self) -> int:
        """Total joints written back to the articulation (7 per arm)."""
        return len(self.controlled_lab_indices)

    # ==================== Setup ====================

    def probe_free_sign(self, arm: str, magnitude: float = 0.5) -> int:
        """Measure which way this arm's elbow can actually swivel, as ``+1`` or ``-1``.

        The redundancy is *one-sided*: each arm tracks ``arm_plane_angle`` freely
        in one direction and saturates within ~0.1 rad in the other, because the
        elbow cannot swing into the torso -- and the two arms are mirrored.
        Saturation is quiet (sub-mm end-effector error, no per-tick jump, no QP
        failure), so it cannot be detected from a single solve's return value.

        The sign is a property of the arm variant's code-generated
        ``arm_plane_angle``, so it is measured here rather than tabulated; a
        table would be silently wrong on ``crsB`` / ``crsP`` / ``acs`` / ``g1``.

        Args:
            arm: Arm name.
            magnitude: Test offset from the home swivel angle, in radians.

        Returns:
            The freely-reachable direction, ``+1`` or ``-1``.
        """
        group = self.ARM_GROUPS[arm]
        home_arm = self.home_model[self.model_indices[arm]].copy()
        home_pose = np.asarray(self._ik.fk7d(group, home_arm).pose, dtype=np.float64)

        tracked: dict[int, float] = {}
        for sign in (-1, 1):
            seed = home_arm.copy()
            for _ in range(40):
                seed = self._solve_raw(arm, seed, home_pose, self.home_apa[arm] + sign * magnitude)
            tracked[sign] = abs(
                wrap_to_pi(float(self._ik.armPlaneAngle(group, seed)) - (self.home_apa[arm] + sign * magnitude))
            )

        self._seed[arm] = home_arm.copy()
        self._last_good[arm] = home_arm.copy()
        self._ik.setModelState(self.home_model)
        return -1 if tracked[-1] <= tracked[1] else 1

    def reset(self, measured_joint_pos_lab: np.ndarray | None = None) -> None:
        """Re-seed every arm and clear the failure-hold state.

        Args:
            measured_joint_pos_lab: Full Lab-order joint vector to re-seed from.
                If ``None``, the projected home pose is used.
        """
        model_q = self.home_model if measured_joint_pos_lab is None else self.jmap.to_model(measured_joint_pos_lab)
        self._ik.setModelState(model_q)
        for arm in self.cfg.arms:
            arm_q = model_q[self.model_indices[arm]].copy()
            self._seed[arm] = arm_q
            self._last_good[arm] = arm_q.copy()
        self.solves = 0
        self.qp_failures = 0

    # ==================== Runtime ====================

    def compute(
        self,
        current_joint_pos_lab: np.ndarray,
        target_poses: np.ndarray,
        swivel_commands: np.ndarray,
    ) -> np.ndarray:
        """Solve every arm for one control tick.

        Args:
            current_joint_pos_lab: Measured joint positions in Lab order, shape
                ``(num_lab_joints,)``.
            target_poses: Desired end-effector poses **in the base-link frame**,
                shape ``(num_arms, 4, 4)``, ordered as :attr:`Ik7dControllerCfg.arms`.
            swivel_commands: One swivel command per arm, shape ``(num_arms,)``,
                interpreted per :attr:`Ik7dControllerCfg.apa_mode`.

        Returns:
            Joint position targets for :attr:`controlled_lab_indices`, in that
            order, shape ``(num_arms * 7,)``.
        """
        model_q = self.jmap.to_model(current_joint_pos_lab)

        # Seed the *whole* model, not just the arm: ``fk7d``/``ik7d`` take only the
        # group's 7 joints and compose them against the model state, so a stale
        # waist silently offsets every pose (0.4 rad of waist is ~0.38 m of hand).
        self._ik.setModelState(model_q)

        solutions = []
        for arm_index, arm in enumerate(self.cfg.arms):
            seed = model_q[self.model_indices[arm]].copy() if self.cfg.seed_from_measured else self._seed[arm]
            apa = self._resolve_swivel(arm, seed, float(swivel_commands[arm_index]))
            q = self._solve_raw(arm, seed, np.asarray(target_poses[arm_index], dtype=np.float64), apa)

            # Failure is out of band: ik_7d returns the seed unchanged, which is
            # indistinguishable from perfect tracking. Check the debug message.
            failed = self._ik.getDebugMsg().sqp_iterations < 0
            self.solves += 1
            if failed:
                self.qp_failures += 1
                self._warn(f"ik_7d QP failed on the {arm} arm; holding the last converged solution")
                if self.cfg.hold_on_failure:
                    q = self._last_good[arm].copy()
            else:
                self._last_good[arm] = q.copy()

            self._seed[arm] = q
            solutions.append(q)

        return np.concatenate(solutions)

    def forward_kinematics(self, arm: str, joint_pos_arm: np.ndarray) -> np.ndarray:
        """End-effector pose in the base-link frame for one arm's joints.

        Composed against the model state set by the last :meth:`compute` or
        :meth:`reset`, so it reflects the current torso configuration.

        Args:
            arm: Arm name.
            joint_pos_arm: The arm's 7 joints, in ik_7d group order.

        Returns:
            A ``(4, 4)`` homogeneous transform.
        """
        return np.asarray(self._ik.fk7d(self.ARM_GROUPS[arm], joint_pos_arm).pose, dtype=np.float64)

    def swivel_angle(self, arm: str, joint_pos_arm: np.ndarray) -> float:
        """Achieved ``arm_plane_angle``, wrapped to ``(-pi, pi]``."""
        return float(self._ik.armPlaneAngle(self.ARM_GROUPS[arm], joint_pos_arm))

    # ==================== Internals ====================

    def _resolve_swivel(self, arm: str, seed: np.ndarray, command: float) -> float:
        """Turn one action element into an absolute ``arm_plane_angle`` command."""
        if self.cfg.apa_mode == "absolute":
            return command
        if self.cfg.apa_mode == "hold":
            return self.swivel_angle(arm, seed)
        return self.home_apa[arm] + self.free_sign[arm] * command

    def _solve_raw(self, arm: str, seed: np.ndarray, target: np.ndarray, apa: float) -> np.ndarray:
        """One ``ik7d`` call, with no bookkeeping. Returns the raw solution."""
        pose = Pose7D()
        # Pose7D's constructor takes no pose keyword; the fields are set after.
        pose.base_frame = self.base_frame[arm]
        pose.frame = self.ee_frame[arm]
        pose.pose = target
        pose.arm_plane_angle = float(apa)
        return np.asarray(self._ik.ik7d(pose, self.ARM_GROUPS[arm], seed, self.cfg.ik_mode), dtype=np.float64)

    def _warn(self, message: str) -> None:
        """Print at most one warning per :attr:`Ik7dControllerCfg.warning_period_s`."""
        if not self.cfg.show_ik_warnings:
            return
        now = time.monotonic()
        if now - self._last_warned < self.cfg.warning_period_s:
            return
        self._last_warned = now
        rate = 100.0 * self.qp_failures / max(self.solves, 1)
        print(f"[ik_7d] {message} ({self.qp_failures}/{self.solves} solves, {rate:.1f}%)")
