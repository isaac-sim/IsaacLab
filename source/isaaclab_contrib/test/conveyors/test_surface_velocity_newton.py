# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native CPU contracts for the Newton conveyor adapter."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBuilderCfg, NewtonCfg, NewtonManager

from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_contrib.conveyors.newton import SurfaceVelocity
from isaaclab_contrib.conveyors.surface_velocity import SurfaceVelocitySpec


def test_native_conveyor_controls_rebind_and_reset_independently(monkeypatch):
    """Real contacts, models and callbacks preserve commands and discard only selected-world traction."""
    cfg = SimulationCfg(
        device="cpu",
        dt=1 / 120,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False), use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=cfg) as sim:
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=cfg.physics))
        for world in range(2):
            builder.begin_world()
            root = f"/World/envs/env_{world}"
            for name, y in (("B", 0.5), ("A", -0.5), ("Nested/BeltA", 2.0)):
                label = f"{root}/{name}" if "/" in name else f"{root}/Belt{name}"
                builder.add_shape_box(
                    -1, xform=wp.transform((0, y, 0), wp.quat_identity()), hx=1, hy=0.2, hz=0.1, label=label
                )
            cube = builder.add_link(xform=wp.transform((0, -0.5, 0.21), wp.quat_identity()), label=f"{root}/Cube0")
            builder.add_shape_box(cube, hx=0.1, hy=0.1, hz=0.1)
            builder.add_articulation([builder.add_joint_free(cube)])
            builder.end_world()
        builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1, label="/World/props/BeltA")
        callbacks_before = set(NewtonManager._callbacks)
        specs = (
            SurfaceVelocitySpec("{ENV_REGEX_NS}/BeltA", velocity=0.1),
            SurfaceVelocitySpec("{ENV_REGEX_NS}/BeltB", velocity=-0.2, enabled=False),
        )
        calls, bindings = [], []

        def actuator():
            calls.append("actuator")

        def force(state):
            calls.append("force")

        def substep(solver, contacts, state, dt):
            calls.append("substep")

        def initialized(model, contacts):
            bindings.append((model, contacts))

        observers = (
            (actuator, NewtonManager.register_post_actuator_callback, NewtonManager.unregister_post_actuator_callback),
            (force, NewtonManager.register_state_force_callback, NewtonManager.unregister_state_force_callback),
            (
                substep,
                NewtonManager.register_post_solver_substep_callback,
                NewtonManager.unregister_post_solver_substep_callback,
            ),
            (initialized, NewtonManager.register_solver_init_callback, NewtonManager.unregister_solver_init_callback),
        )
        for callback, register, _ in observers:
            register(callback)
        NewtonManager.register_solver_init_callback(initialized)
        driver = SurfaceVelocity(2, specs, body_pattern="Cube0$", body_count_per_env=1)
        try:
            with pytest.raises(RuntimeError, match="not bound"):
                driver.set_velocities(0.2)
            sim.reset()
            assert bindings == [(NewtonManager.get_model(), NewtonManager.get_contacts())]
            first = driver._binding
            assert NewtonManager.get_contacts().force is not None
            assert driver.prim_paths == tuple(
                f"/World/envs/env_{world}/Belt{name}" for world in range(2) for name in ("A", "B")
            )
            model = NewtonManager.get_model()
            mapping = first._conveyor.shape_conveyor.numpy()
            assert [
                mapping[i] for i, label in enumerate(model.shape_label) if "Nested" in label or "/props/" in label
            ] == [-1] * 3
            driver.set_velocities([0.3, -0.25, -0.4], indices=[0, 1, 3])
            driver.set_enabled([False, True], indices=[2, 3])
            np.testing.assert_allclose(driver.get_velocities().numpy(), [0.3, 0, 0, -0.4])
            sim.step(render=False)
            assert calls == ["actuator"] + ["force", "substep"] * cfg.physics.num_substeps
            np.testing.assert_allclose(driver.get_encoder_positions().numpy(), np.array([0.3, 0, 0, -0.4]) / 120)
            first._conveyor.conveyor_body_f.assign(np.ones((2, 6), dtype=np.float32))
            driver.reset([0])
            state = model.state()
            first.apply(state)
            np.testing.assert_array_equal(state.body_f.numpy(), [[0] * 6, [1] * 6])
            np.testing.assert_allclose(driver.get_encoder_positions().numpy(), [0, 0, 0, -0.4 / 120])
            sim.reset()
            assert first._closed and driver._binding is not first
            assert driver._binding._model is NewtonManager.get_model()
            assert driver._binding._contacts is NewtonManager.get_contacts()
            np.testing.assert_allclose(driver.get_velocities().numpy(), [0.3, 0, 0, -0.4])
            assert len(bindings) == 2 and bindings[1] == (NewtonManager.get_model(), NewtonManager.get_contacts())
            assert bindings[0][0] is not bindings[1][0] and bindings[0][1] is not bindings[1][1]
            for callback, _, unregister in observers:
                unregister(callback)
                unregister(callback)
            previous_calls = list(calls)
            sim.step(render=False)
            sim.reset()
            assert calls == previous_calls and len(bindings) == 2
            np.testing.assert_array_equal(driver.get_enabled().numpy(), [1, 0, 0, 1])
            np.testing.assert_allclose(driver.get_commanded_velocities().numpy(), [0.3, -0.25, 0.1, -0.4])
            callbacks_live = set(NewtonManager._callbacks)
            with monkeypatch.context() as patch:

                def fail_registration(cls, callback):
                    raise RuntimeError("solver callback unavailable")

                patch.setattr(NewtonManager, "register_solver_init_callback", classmethod(fail_registration))
                with pytest.raises(RuntimeError, match="solver callback unavailable"):
                    SurfaceVelocity(2, specs, body_pattern="Cube0$")
            assert set(NewtonManager._callbacks) == callbacks_live
        finally:
            driver.close()
            driver.close()
        assert set(NewtonManager._callbacks) == callbacks_before
        assert not NewtonManager._solver_init_callbacks
        assert not NewtonManager._state_force_callbacks
        assert not NewtonManager._post_solver_substep_callbacks


@pytest.mark.parametrize(
    ("specs", "num_envs", "env_path", "error", "message"),
    [
        ((), 1, "/World/envs/env_{}", ValueError, "At least one"),
        ((object(),), 1, "/World/envs/env_{}", TypeError, "SurfaceVelocitySpec"),
        ((SurfaceVelocitySpec("/Belt"),) * 2, 1, "/World/envs/env_{}", ValueError, "unique"),
        (
            (SurfaceVelocitySpec("/Belt"), SurfaceVelocitySpec("/Belt/Child")),
            1,
            "/World/envs/env_{}",
            ValueError,
            "ancestors",
        ),
        ((SurfaceVelocitySpec("/Curve", curved=True),), 1, "/World/envs/env_{}", ValueError, "positive radius"),
        ((SurfaceVelocitySpec("/Shared/Belt"),), 2, "/World/envs/env_{}", ValueError, "Replicated"),
        ((SurfaceVelocitySpec("{ENV_REGEX_NS}/Belt"),), 1, "/World/envs/env_.*", ValueError, "env_path_format"),
    ],
)
def test_invalid_conveyors_leave_no_callbacks(specs, num_envs, env_path, error, message):
    """Rejected authoring must not leave a partially registered adapter."""
    before = set(NewtonManager._callbacks)
    with pytest.raises(error, match=message):
        SurfaceVelocity(num_envs, specs, env_path_format=env_path, body_pattern="Cube0$")
    assert set(NewtonManager._callbacks) == before


def test_unreplicated_absolute_surface_path_is_preserved():
    """A single-environment surface can target a global authored path."""
    from isaaclab_contrib.conveyors.newton import _resolve_belt_prim_path

    assert _resolve_belt_prim_path("/World/Shared/Belt", "/World/envs/env_{}", 0) == "/World/Shared/Belt"
