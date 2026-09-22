# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-agnostic helpers for Lab-vs-Newton actuator equivalence tests.

Shared between the Newton-backend and PhysX-backend actuator test twins:

* the actuator configuration dictionaries under test (ANYmal-C leg groups
  plus the cartpole explicit group),
* :class:`EquivalenceAssertionsMixin` with the trajectory/telemetry
  ``test_*_match`` oracles,
* dummy TorchScript checkpoint factories for the neural actuator tests,
* mock scene/env plumbing for driving ``randomize_actuator_gains``,
* :class:`ActuatorStateResetBase` with the per-env actuator state reset
  scenario.

This module must stay free of backend packages (``isaaclab_newton``,
``isaaclab_physx``) and of ``isaaclab_assets``; everything backend-specific
is injected by the twin test files through subclass hooks.
"""

import json
import tempfile

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

# ---------------------------------------------------------------------------
# Actuator configurations under test
# ---------------------------------------------------------------------------

IDEAL_PD_ACTUATORS = {
    "legs": IdealPDActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
        actuator_effort_limit=80.0,
    ),
}

DC_MOTOR_ACTUATORS = {
    "legs": DCMotorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        saturation_effort=120.0,
        actuator_effort_limit=80.0,
        actuator_velocity_limit=7.5,
        stiffness={".*": 40.0},
        damping={".*": 5.0},
    ),
}

MIXED_ACTUATORS = {
    "hips": IdealPDActuatorCfg(
        joint_names_expr=[".*HAA"],
        stiffness=40.0,
        damping=5.0,
        actuator_effort_limit=80.0,
    ),
    "knees": DCMotorCfg(
        joint_names_expr=[".*HFE", ".*KFE"],
        saturation_effort=120.0,
        actuator_effort_limit=80.0,
        actuator_velocity_limit=7.5,
        stiffness={".*": 40.0},
        damping={".*": 5.0},
    ),
}

MIXED_WITH_IMPLICIT_ACTUATORS = {
    "hips": ImplicitActuatorCfg(
        joint_names_expr=[".*HAA"],
        stiffness=40.0,
        damping=5.0,
    ),
    "thighs": IdealPDActuatorCfg(
        joint_names_expr=[".*HFE"],
        stiffness=40.0,
        damping=5.0,
        actuator_effort_limit=80.0,
    ),
    "knees": DCMotorCfg(
        joint_names_expr=[".*KFE"],
        saturation_effort=120.0,
        actuator_effort_limit=80.0,
        actuator_velocity_limit=7.5,
        stiffness=40.0,
        damping=5.0,
    ),
}

DELAYED_PD_ACTUATORS = {
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
        actuator_effort_limit=80.0,
        min_delay=2,
        max_delay=4,
    ),
}

IMPLICIT_ONLY_ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        stiffness=40.0,
        damping=5.0,
    ),
}

CARTPOLE_EXPLICIT_ACTUATORS = {
    "all_joints": IdealPDActuatorCfg(
        joint_names_expr=["slider_to_cart", "cart_to_pole"],
        stiffness=10.0,
        damping=1.0,
        actuator_effort_limit=100.0,
    ),
}

# ---------------------------------------------------------------------------
# Equivalence assertions
# ---------------------------------------------------------------------------


class EquivalenceAssertionsMixin:
    """Trajectory/telemetry oracles shared by the Lab-vs-Newton equivalence bases.

    Mixed into each twin's ``_EquivalenceTestBase`` (alongside
    ``unittest.TestCase``). The twin's ``setUpClass`` must populate
    ``cls.lab_result`` and ``cls.newton_result`` with the dictionaries
    returned by its backend-specific simulation runner.
    """

    pos_atol: float = 2e-3
    pos_rtol: float = 1e-3
    vel_atol: float = 1e-2
    vel_rtol: float = 1e-2
    torque_atol: float = 1e-3
    torque_rtol: float = 1e-3

    def test_joint_positions_match(self):
        for step_i, (lab, newton) in enumerate(zip(self.lab_result["joint_pos"], self.newton_result["joint_pos"])):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.pos_atol,
                rtol=self.pos_rtol,
                msg=f"Joint positions diverged at step {step_i}",
            )

    def test_joint_velocities_match(self):
        for step_i, (lab, newton) in enumerate(zip(self.lab_result["joint_vel"], self.newton_result["joint_vel"])):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.vel_atol,
                rtol=self.vel_rtol,
                msg=f"Joint velocities diverged at step {step_i}",
            )

    def test_applied_effort_match(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["applied_effort"], self.newton_result["applied_effort"])
        ):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.torque_atol,
                rtol=self.torque_rtol,
                msg=f"applied_effort diverged at step {step_i}",
            )

    def test_computed_effort_match(self):
        for step_i, (lab, newton) in enumerate(
            zip(self.lab_result["computed_effort"], self.newton_result["computed_effort"])
        ):
            torch.testing.assert_close(
                lab,
                newton,
                atol=self.torque_atol,
                rtol=self.torque_rtol,
                msg=f"computed_effort diverged at step {step_i}",
            )


# ---------------------------------------------------------------------------
# Neural network actuator checkpoints
# ---------------------------------------------------------------------------


def make_dummy_mlp_checkpoint(device: str = "cpu") -> str:
    """Create a minimal TorchScript MLP checkpoint with metadata.

    The network accepts 6 inputs (3 history steps x 2 features per step
    in pos_vel order) and outputs 1 effort.
    """
    torch.manual_seed(42)
    net = (
        torch.nn.Sequential(
            torch.nn.Linear(6, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 1),
        )
        .to(device)
        .eval()
    )
    scripted = torch.jit.script(net)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    extra = {
        "metadata.json": json.dumps(
            {
                "model_type": "mlp",
                "input_order": "pos_vel",
                "input_idx": [0, 1, 2],
                "pos_scale": 1.0,
                "vel_scale": 0.5,
                "torque_scale": 2.0,
            }
        )
    }
    torch.jit.save(scripted, tmp_path, _extra_files=extra)
    return tmp_path


class _DummyLSTM(torch.nn.Module):
    """Minimal LSTM network for actuator testing."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(4, 1)

    def forward(
        self,
        x: torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hc_new = self.lstm(x, hc)
        return self.fc(out[:, -1, :]), hc_new


def make_dummy_lstm_checkpoint(device: str = "cpu") -> str:
    """Create a minimal TorchScript LSTM checkpoint with metadata."""
    torch.manual_seed(42)
    net = _DummyLSTM().to(device).eval()
    scripted = torch.jit.script(net)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    extra = {"metadata.json": json.dumps({"model_type": "lstm"})}
    torch.jit.save(scripted, tmp_path, _extra_files=extra)
    return tmp_path


# ---------------------------------------------------------------------------
# Domain randomization mocks
# ---------------------------------------------------------------------------


class MockScene:
    """Minimal stand-in for ``InteractiveScene`` accepted by ``ManagerTermBase``."""

    def __init__(self, assets: dict, num_envs: int):
        self._assets = assets
        self.num_envs = num_envs

    def __getitem__(self, name: str):
        return self._assets[name]


class MockEnv:
    """Minimal stand-in for ``ManagerBasedEnv`` for invoking DR terms.

    ``randomize_actuator_gains`` only reads ``env.scene[name]`` and
    ``env.scene.num_envs`` (plus ``env.num_envs`` / ``env.device`` from the
    ``ManagerTermBase`` properties). No simulator access is needed because
    the DR term reaches the actuator adapter through the articulation.
    """

    def __init__(self, assets: dict, num_envs: int, device: str):
        self.scene = MockScene(assets, num_envs)
        self.num_envs = num_envs
        self.device = device


def build_dr_term(env, asset_name, joint_ids=None):
    """Build a ``randomize_actuator_gains`` event term bound to ``asset_name``."""
    from isaaclab.envs.mdp.events import randomize_actuator_gains  # noqa: PLC0415
    from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: PLC0415

    asset_cfg = SceneEntityCfg(asset_name)
    if joint_ids is not None:
        asset_cfg.joint_ids = joint_ids
    cfg = EventTermCfg(
        func=randomize_actuator_gains,
        params={
            "asset_cfg": asset_cfg,
            "stiffness_distribution_params": (100.0, 100.0),
            "damping_distribution_params": (5.0, 5.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    return randomize_actuator_gains(cfg, env), asset_cfg


# ---------------------------------------------------------------------------
# Per-env reset: actuator state isolation
# ---------------------------------------------------------------------------


class ActuatorStateResetBase:
    """Reset must clear the actuator state buffers for the requested envs only.

    Inspects ``adapter.actuators[i].state.delay_state.num_pushes`` directly:

    * After warmup, ``num_pushes > 0`` for every DOF (buffer was populated).
    * After ``articulation.reset(env_ids=[0])``, the entries for env 0's DOFs
      must be ``0`` and the entries for env 1's DOFs must remain ``> 0``.

    Done independently on Lab and Newton paths. Backend-specific twins mix
    this into a ``unittest.TestCase`` and provide :meth:`_make_sim_cfg`,
    :meth:`_make_articulation`, and :meth:`_get_adapter`.
    """

    RESET_ENV: int = 0
    UNCHANGED_ENV: int = 1
    NUM_ENVS: int = 2
    DT: float = 1.0 / 120.0
    TARGET_OFFSET: float = 0.1  # [rad] added to initial joint positions
    RESET_WARMUP_STEPS: int = 3

    def _make_sim_cfg(self, use_newton_actuators: bool) -> SimulationCfg:
        """Return the backend simulation config for the run."""
        raise NotImplementedError

    def _make_articulation(self):
        """Construct the backend articulation (DelayedPD on all joints) at ``/World/Env_.*/Robot``."""
        raise NotImplementedError

    def _get_adapter(self, articulation):
        """Return the Newton actuator adapter that owns ``articulation``'s actuators."""
        raise NotImplementedError

    def _build_and_warm(self, *, use_newton_actuators: bool):
        ctx = build_simulation_context(
            device="cuda:0",
            gravity_enabled=True,
            add_ground_plane=True,
            sim_cfg=self._make_sim_cfg(use_newton_actuators),
        )
        sim = ctx.__enter__()
        sim._app_control_on_stop_handle = None
        for i in range(self.NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 3.0, 0, 0))
        articulation = self._make_articulation()
        sim.reset()

        init_pos = wp.to_torch(articulation.data.joint_pos).clone()
        target_pos = init_pos + self.TARGET_OFFSET
        target_vel = torch.zeros_like(init_pos)
        articulation.set_joint_position_target_index(target=target_pos)
        articulation.set_joint_velocity_target_index(target=target_vel)
        for _ in range(self.RESET_WARMUP_STEPS):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(self.DT)
        return ctx, sim, articulation

    def test_newton_state_reset_isolated_to_reset_env(self):
        """Newton: ``num_pushes`` zeroes for env 0's DOFs only after reset of [0]."""
        ctx, sim, articulation = self._build_and_warm(use_newton_actuators=True)
        try:
            adapter = self._get_adapter(articulation)
            self.assertIsNotNone(adapter)
            # Find a DelayedPD actuator (it's the only one with delay_state).
            stateful_pairs = [
                (act, st)
                for act, st in zip(adapter.actuators, adapter._states_a)
                if st is not None and getattr(st, "delay_state", None) is not None
            ]
            self.assertGreater(len(stateful_pairs), 0, "expected at least one DelayedPD actuator with delay_state")

            for act, state in stateful_pairs:
                pushes_before = state.delay_state.num_pushes.numpy()
                self.assertTrue(
                    (pushes_before > 0).all(),
                    "expected non-zero num_pushes for all DOFs after warmup",
                )

            articulation.reset(env_ids=torch.tensor([self.RESET_ENV], device=articulation.device, dtype=torch.long))

            # Map each entry of ``act.indices`` to its env via the adapter's
            # per-env DOF count. On the Newton backend the adapter is model-wide
            # (includes free-joint DOFs on floating-base articulations); on
            # PhysX it is per-articulation — ``adapter.num_joints`` is the
            # correct stride in both cases.
            for act, state in stateful_pairs:
                pushes_after = state.delay_state.num_pushes.numpy()
                indices_np = act.indices.numpy()
                for i, global_dof in enumerate(indices_np):
                    env = int(global_dof) // adapter.num_joints
                    if env == self.RESET_ENV:
                        self.assertEqual(
                            int(pushes_after[i]),
                            0,
                            f"DOF {i} (env {env}) should be reset to 0, got {pushes_after[i]}",
                        )
                    else:
                        self.assertGreater(
                            int(pushes_after[i]),
                            0,
                            f"DOF {i} (env {env}) was NOT in reset env_ids but num_pushes is 0",
                        )
        finally:
            ctx.__exit__(None, None, None)

    def test_lab_state_reset_isolated_to_reset_env(self):
        """Lab: DelayedPDActuator circular buffer zeroed for env 0 only."""
        ctx, sim, articulation = self._build_and_warm(use_newton_actuators=False)
        try:
            from isaaclab.actuators import DelayedPDActuator  # noqa: PLC0415

            delayed = [a for a in articulation.actuators.values() if isinstance(a, DelayedPDActuator)]
            self.assertGreater(len(delayed), 0, "expected at least one Lab DelayedPDActuator")
            actuator = delayed[0]
            buf = actuator.positions_delay_buffer._circular_buffer._buffer
            # ``_buffer`` shape: (max_length, batch_size, num_joints).
            self.assertIsNotNone(buf, "delay buffer should be populated after warmup")
            self.assertTrue(
                (buf[:, self.UNCHANGED_ENV] != 0).any().item(),
                "expected non-zero buffer entries for env 1 after warmup",
            )

            articulation.reset(env_ids=torch.tensor([self.RESET_ENV], device=articulation.device, dtype=torch.long))

            self.assertTrue(
                torch.all(buf[:, self.RESET_ENV] == 0).item(),
                f"Lab: env {self.RESET_ENV} buffer not zeroed after reset.",
            )
            self.assertTrue(
                (buf[:, self.UNCHANGED_ENV] != 0).any().item(),
                f"Lab: env {self.UNCHANGED_ENV} buffer was zeroed — reset leaked into an unselected env.",
            )
        finally:
            ctx.__exit__(None, None, None)
