# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reusable unified reinforcement learning entrypoints."""

from __future__ import annotations

import importlib
import runpy
import subprocess
import sys
import types
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab_rl.entrypoints import PlaybackRequest, TrainingRequest, api, dispatch
from isaaclab_rl.entrypoints import simple_agents as _simple_agents
from isaaclab_rl.entrypoints.simple_agents import _create_zero_action_policy


@pytest.mark.parametrize(
    ("statement", "unexpected_modules"),
    [
        ("import isaaclab_rl", ["isaaclab_rl.entrypoints", "torch"]),
        ("import isaaclab_rl.entrypoints", ["isaaclab_rl.entrypoints.multigpu", "torch"]),
        ("import isaaclab_rl.entrypoints.backends", ["torch"]),
        ("import isaaclab_rl.rl_games", ["isaaclab_rl.rl_games.rl_games", "rl_games", "torch"]),
        ("import isaaclab_rl.rl_games.pbt", ["isaaclab_rl.rl_games.pbt.pbt", "rl_games", "torch"]),
        ("import isaaclab_rl.rsl_rl", ["isaaclab_rl.rsl_rl.vecenv_wrapper", "rsl_rl", "torch"]),
        ("import isaaclab_rl.utils", ["torch"]),
        (
            "from isaaclab_rl import run_play_cli",
            ["isaaclab_rl.entrypoints.multigpu", "torch"],
        ),
        (
            "from isaaclab_rl.entrypoints import run_play_cli",
            ["isaaclab_rl.entrypoints.multigpu", "torch"],
        ),
    ],
)
def test_public_namespaces_do_not_import_unrelated_frameworks(statement: str, unexpected_modules: list[str]) -> None:
    """Importing public namespaces must not eagerly load unrelated RL frameworks."""
    unexpected_modules_literal = repr(unexpected_modules)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"{statement}; import sys; "
            f"unexpected = [name for name in {unexpected_modules_literal} if name in sys.modules]; "
            "assert not unexpected, f'Unexpected eager imports: {unexpected}'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_zero_agent_infers_finite_manager_actions() -> None:
    """The zero agent holds absolute task-space targets and zeros all other action terms."""

    class DifferentialInverseKinematicsAction:
        action_dim = 7
        cfg = SimpleNamespace(controller=SimpleNamespace(use_relative_mode=False, command_type="pose"))
        _scale = torch.tensor([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

        def _compute_frame_pose(self):
            return torch.tensor([[2.0, 4.0, 6.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    class RMPFlowAction:
        action_dim = 7
        cfg = SimpleNamespace(use_relative_mode=False)
        _scale = torch.ones(7)

        def _compute_frame_pose(self):
            return torch.tensor([[3.0, 2.0, 1.0]]), torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    class PinkInverseKinematicsAction:
        action_dim = 16
        cfg = SimpleNamespace(target_eef_link_names={"left": "left_hand", "right": "right_hand"})
        _hand_joint_ids = [1, 3]
        _num_frame_tasks = 2

        def __init__(self):
            body_poses = torch.tensor(
                [
                    [
                        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                        [4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0],
                    ]
                ]
            )
            self._asset = SimpleNamespace(
                find_bodies=lambda expressions, preserve_order: ([1, 0], ["left_hand", "right_hand"]),
                data=SimpleNamespace(
                    body_link_pose_w=SimpleNamespace(torch=body_poses),
                    joint_pos=SimpleNamespace(torch=torch.tensor([[0.1, 0.2, 0.3, 0.4]])),
                ),
            )

    class OperationalSpaceControllerAction:
        action_dim = 7
        raw_actions = torch.zeros(1, 7)
        _pose_abs_idx = 0
        _position_scale = torch.ones(3)
        _orientation_scale = torch.ones(4)
        _task_frame_pose_b = None
        _ee_pose_b = torch.tensor([[9.0, 8.0, 7.0, 0.0, 1.0, 0.0, 0.0]])

        def _compute_ee_pose(self):
            pass

        def _compute_task_frame_pose(self):
            pass

    class JointAction:
        action_dim = 2

    terms = {
        "diff_ik": DifferentialInverseKinematicsAction(),
        "rmpflow": RMPFlowAction(),
        "pink": PinkInverseKinematicsAction(),
        "osc": OperationalSpaceControllerAction(),
        "joints": JointAction(),
    }
    manager = SimpleNamespace(
        action=torch.empty(1, sum(term.action_dim for term in terms.values())),
        active_terms=list(terms),
        get_term=terms.__getitem__,
    )
    unwrapped = SimpleNamespace(
        action_manager=manager,
        scene=SimpleNamespace(env_origins=torch.tensor([[1.0, 1.0, 1.0]])),
    )

    actions = _create_zero_action_policy(SimpleNamespace(unwrapped=unwrapped))()

    expected_pink_poses = torch.tensor(
        [[[3.0, 4.0, 5.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0]]]
    ).flatten(start_dim=1)
    expected = torch.cat(
        (
            torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0]]),
            expected_pink_poses,
            torch.tensor([[0.2, 0.4]]),
            OperationalSpaceControllerAction._ee_pose_b,
            torch.zeros(1, 2),
        ),
        dim=-1,
    )

    assert torch.equal(actions, expected)
    assert torch.isfinite(actions).all()


def test_zero_agent_rejects_non_finite_inferred_actions() -> None:
    """The zero agent stops before a non-finite inferred action reaches the environment."""

    class DifferentialInverseKinematicsAction:
        action_dim = 7
        cfg = SimpleNamespace(controller=SimpleNamespace(use_relative_mode=False, command_type="pose"))
        _scale = torch.ones(7)

        def _compute_frame_pose(self):
            return torch.full((1, 3), torch.nan), torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    term = DifferentialInverseKinematicsAction()
    manager = SimpleNamespace(
        action=torch.empty(1, term.action_dim),
        active_terms=["ik"],
        get_term=lambda name: term,
    )
    unwrapped = SimpleNamespace(action_manager=manager)
    policy = _create_zero_action_policy(SimpleNamespace(unwrapped=unwrapped))

    with pytest.raises(RuntimeError, match="inferred non-finite actions"):
        policy()


def test_zero_agent_supports_composite_direct_action_spaces() -> None:
    """Direct environments receive tensorized zeros matching composite action spaces."""
    action_space = gym.spaces.Dict(
        {
            "continuous": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "discrete": gym.spaces.Discrete(3),
        }
    )
    unwrapped = SimpleNamespace(
        action_manager=None,
        single_action_space=action_space,
        device="cpu",
        num_envs=2,
    )

    actions = _create_zero_action_policy(SimpleNamespace(unwrapped=unwrapped))()

    assert torch.equal(actions["continuous"], torch.zeros(2, 2))
    assert torch.equal(actions["discrete"], torch.zeros(2, 1, dtype=torch.int64))


def test_zero_agent_supports_direct_multi_agent_action_spaces() -> None:
    """Direct multi-agent environments receive a zero action for every agent."""
    unwrapped = SimpleNamespace(
        action_manager=None,
        action_spaces={
            "robot": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "object": gym.spaces.Discrete(2),
        },
        device="cpu",
        num_envs=3,
    )

    actions = _create_zero_action_policy(SimpleNamespace(unwrapped=unwrapped))()

    assert torch.equal(actions["robot"], torch.zeros(3, 2))
    assert torch.equal(actions["object"], torch.zeros(3, 1, dtype=torch.int64))


def test_zero_agent_rejects_invalid_config_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported task presets fail cleanly before a simulator backend is initialized."""

    class _InvalidCfg:
        scene = SimpleNamespace(num_envs=1)
        sim = SimpleNamespace(device="cpu", use_fabric=True)

        def validate(self) -> None:
            raise ValueError("unsupported physics backend")

    args = SimpleNamespace(num_envs=None, device=None, disable_fabric=False, task="Invalid-Task")
    monkeypatch.setattr(_simple_agents, "_parse_args", lambda argv, policy: args)
    monkeypatch.setattr(_simple_agents, "resolve_task_config", lambda task, agent: (_InvalidCfg(), None))
    monkeypatch.setattr(
        _simple_agents,
        "launch_simulation",
        lambda *args, **kwargs: pytest.fail("simulation launched before config validation"),
    )

    with pytest.raises(SystemExit, match="Invalid environment configuration: unsupported physics backend"):
        _simple_agents.run([], policy="zero")


def test_train_request_adapts_typed_parameters_to_cli(monkeypatch) -> None:
    """Training requests use typed parameters rather than parser namespaces."""
    received: list[str] = []

    def fake_run_train_cli(argv: list[str]) -> int:
        received.extend(argv)
        return 7

    monkeypatch.setattr(api, "run_train_cli", fake_run_train_cli)

    result = api.train(
        TrainingRequest(
            backend="rsl_rl",
            task="Isaac-Cartpole",
            num_envs=32,
            max_iterations=10,
            distributed=True,
            hydra_args=("physics=newton_mjwarp",),
        )
    )

    assert result == 7
    assert received == [
        "--rl_library",
        "rsl_rl",
        "--task",
        "Isaac-Cartpole",
        "--num_envs",
        "32",
        "--max_iterations",
        "10",
        "--distributed",
        "physics=newton_mjwarp",
    ]


def test_train_request_maps_checkpoint_to_backend_argument(monkeypatch) -> None:
    """Training requests forward a checkpoint so training can resume from it."""
    received: list[str] = []

    def fake_run_train_cli(argv: list[str]) -> int:
        received.extend(argv)
        return 0

    monkeypatch.setattr(api, "run_train_cli", fake_run_train_cli)

    api.train(TrainingRequest(backend="rsl_rl", task="Isaac-Cartpole", checkpoint="latest"))
    assert received == ["--rl_library", "rsl_rl", "--task", "Isaac-Cartpole", "--checkpoint", "latest"]

    received.clear()
    api.train(TrainingRequest(backend="rlinf", task="Isaac-Task", checkpoint="model"))
    assert received == ["--rl_library", "rlinf", "--task", "Isaac-Task", "--checkpoint", "model"]


def test_rlinf_parser_uses_unified_checkpoint_and_iteration_flags() -> None:
    """RLinf accepts the public checkpoint and iteration option names."""
    from isaaclab_rl.entrypoints.backends import train_rlinf

    args = train_rlinf._parse_args(["--config_name", "ppo", "--checkpoint", "latest", "--max_iterations", "10"])

    assert args.checkpoint == "latest"
    assert args.max_iterations == 10


def test_rlinf_rejects_pretrained_checkpoint() -> None:
    """RLinf has no published pre-trained checkpoint."""
    from isaaclab_rl.entrypoints.backends.cli_args_rlinf import _resolve_rlinf_checkpoint

    with pytest.raises(ValueError, match="Pre-trained checkpoints are not available for RLinf"):
        _resolve_rlinf_checkpoint("pretrained", log_root_path="logs/rlinf", task="Isaac-Task", config_name="ppo")


def test_run_backend_restores_sys_argv_after_training(monkeypatch) -> None:
    """A training backend that mutates ``sys.argv`` must not leak the change to the caller."""
    module = types.ModuleType("fake_train_backend")

    def run(argv: list[str]) -> None:
        # backends call set_hydra_args, which overwrites sys.argv while parsing
        sys.argv = [sys.argv[0]] + argv

    module.run = run
    monkeypatch.setattr(importlib, "import_module", lambda name: module)

    original_argv = list(sys.argv)
    dispatch._run_backend("fake_train_backend", ["physics=newton_mjwarp"], run_as_script=False)

    assert sys.argv == original_argv


def test_play_request_uses_unified_checkpoint_argument(monkeypatch) -> None:
    """RLinf requests map shared fields to its focused backend arguments."""
    received: list[str] = []

    def fake_run_play_cli(argv: list[str]) -> int:
        received.extend(argv)
        return 0

    monkeypatch.setattr(api, "run_play_cli", fake_run_play_cli)

    api.play(PlaybackRequest(backend="rlinf", task="Isaac-Task", checkpoint="model", video=True))

    assert received == [
        "--rl_library",
        "rlinf",
        "--task",
        "Isaac-Task",
        "--checkpoint",
        "model",
        "--video",
    ]


def test_train_dispatches_selected_backend(monkeypatch) -> None:
    """The unified training dispatcher forwards only backend arguments."""
    received: dict[str, object] = {}

    def _fake_run_backend(module_name: str, argv: list[str], *, run_as_script: bool) -> None:
        received.update(module_name=module_name, argv=argv, run_as_script=run_as_script)

    monkeypatch.setattr(dispatch, "_run_backend", _fake_run_backend)

    assert dispatch.run_train_cli(["--rl_library", "rsl_rl", "--task", "Isaac-Cartpole"]) == 0
    assert received == {
        "module_name": "isaaclab_rl.entrypoints.backends.train_rsl_rl",
        "argv": ["--task", "Isaac-Cartpole"],
        "run_as_script": False,
    }


def test_dispatch_uses_task_registered_default_backend(monkeypatch) -> None:
    """A task registry default selects the backend when the CLI omits it."""
    task_name = "Isaac-DefaultAgentDispatchTest"
    gym.register(id=task_name, entry_point="dummy:Env", kwargs={"default_agent": "rsl_rl"})
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", types.ModuleType("isaaclab_tasks"))
    received: dict[str, object] = {}
    monkeypatch.setattr(
        dispatch,
        "_run_backend",
        lambda module_name, argv, *, run_as_script: received.update(
            module_name=module_name, argv=argv, run_as_script=run_as_script
        ),
    )

    try:
        assert dispatch.run_train_cli(["--task", task_name]) == 0
    finally:
        gym.registry.pop(task_name, None)

    assert received == {
        "module_name": "isaaclab_rl.entrypoints.backends.train_rsl_rl",
        "argv": ["--task", task_name],
        "run_as_script": False,
    }


def test_dispatch_fuses_option_like_kit_args(monkeypatch) -> None:
    """Space-separated option-like Kit arguments are fused before backend parsing."""
    received: dict[str, object] = {}
    monkeypatch.setattr(
        dispatch, "_run_backend", lambda module_name, argv, *, run_as_script: received.update(argv=argv)
    )

    dispatch.run_train_cli(["--rl_library", "rsl_rl", "--kit_args", "--foo=/bar"])

    assert received["argv"] == ["--kit_args=--foo=/bar"]


def test_dispatch_requires_a_backend() -> None:
    """Missing backend selection returns the conventional CLI error status."""
    assert dispatch.run_train_cli([]) == 2


def _torch_backend_state() -> tuple[bool, bool, bool, bool]:
    import torch

    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )


def test_scoped_backend_state_restores_values_after_exception() -> None:
    """Backend-global settings are restored when a scoped operation fails."""
    from isaaclab_rl.entrypoints.common import preserve_attribute, scoped_torch_backend_flags

    original = _torch_backend_state()
    holder = types.SimpleNamespace(value="original")

    with pytest.raises(RuntimeError, match="failed"):
        with (
            scoped_torch_backend_flags(
                cuda_matmul_allow_tf32=False,
                cudnn_allow_tf32=True,
                cudnn_deterministic=True,
                cudnn_benchmark=False,
            ),
            preserve_attribute(holder, "value"),
        ):
            holder.value = "temporary"
            assert _torch_backend_state() == (False, True, True, False)
            raise RuntimeError("failed")

    assert _torch_backend_state() == original
    assert holder.value == "original"


def test_rejected_rsl_training_preserves_torch_backend_state(monkeypatch) -> None:
    """A rejected in-process RSL-RL request does not mutate its caller."""
    import torch

    caller_state = (False, False, True, True)
    settings = (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
    )
    for (target, name), value in zip(settings, caller_state):
        monkeypatch.setattr(target, name, value)

    with pytest.raises(SystemExit):
        dispatch._run_backend("isaaclab_rl.entrypoints.backends.train_rsl_rl", ["--help"], run_as_script=False)

    assert _torch_backend_state() == caller_state


def test_failed_rsl_training_restores_torch_backend_state(monkeypatch) -> None:
    """RSL-RL training restores its caller's Torch settings after a failure."""
    import torch

    from isaaclab_rl.entrypoints.backends import train_rsl_rl

    caller_state = (False, False, True, True)
    settings = (
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "benchmark"),
    )
    for (target, name), value in zip(settings, caller_state):
        monkeypatch.setattr(target, name, value)

    monkeypatch.setattr(train_rsl_rl, "_parse_args", lambda argv: types.SimpleNamespace())

    def fail_after_mutation(_args_cli) -> None:
        assert _torch_backend_state() == (True, True, False, False)
        raise RuntimeError("failed")

    monkeypatch.setattr(train_rsl_rl, "_run", fail_after_mutation)
    with pytest.raises(RuntimeError, match="failed"):
        train_rsl_rl.run([])

    assert _torch_backend_state() == caller_state


def test_rsl_training_registers_external_task_before_agent_discovery(monkeypatch) -> None:
    """RSL-RL parses tasks registered by its external callback."""
    from isaaclab_rl.entrypoints.backends import train_rsl_rl

    task_name = "Isaac-ExternalCallbackOrderTest"
    callback_module_name = "_external_callback_order_test"
    callback_module = types.ModuleType(callback_module_name)

    def register_task() -> list[str]:
        gym.register(
            id=task_name,
            entry_point="dummy:Env",
            kwargs={"rsl_rl_cfg_entry_point": "dummy:AgentCfg"},
        )
        return []

    callback_module.register_task = register_task
    monkeypatch.setitem(sys.modules, callback_module_name, callback_module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--task", task_name, "--external_callback", f"{callback_module_name}.register_task"],
    )
    gym.registry.pop(task_name, None)

    try:
        args = train_rsl_rl._parse_args(sys.argv[1:])
    finally:
        gym.registry.pop(task_name, None)

    assert args.task == task_name


def test_skrl_training_restores_jax_backend(monkeypatch) -> None:
    """SKRL training removes the JAX backend setting it created after an exception."""
    skrl = pytest.importorskip("skrl")

    from isaaclab_rl.entrypoints.backends import train_skrl

    monkeypatch.delattr(skrl.config.jax, "backend", raising=False)
    monkeypatch.setattr(train_skrl, "_parse_args", lambda argv: types.SimpleNamespace(ml_framework="jax"))

    def fail_after_mutation(_args_cli) -> None:
        assert skrl.config.jax.backend == "jax"
        skrl.config.jax.backend = "mutated"
        raise RuntimeError("failed")

    monkeypatch.setattr(train_skrl, "_run", fail_after_mutation)
    with pytest.raises(RuntimeError, match="failed"):
        train_skrl.run([])

    assert not hasattr(skrl.config.jax, "backend")


def test_skrl_play_main_restores_jax_backend(monkeypatch) -> None:
    """Direct SKRL play calls remove the JAX backend setting they created."""
    pytest.importorskip("skrl")
    monkeypatch.setattr(sys, "argv", ["play_skrl.py"])
    namespace = runpy.run_module("isaaclab_rl.entrypoints.backends.play_skrl", run_name="test_play_skrl")
    skrl = namespace["skrl"]
    namespace["args_cli"].ml_framework = "jax"
    monkeypatch.delattr(skrl.config.jax, "backend", raising=False)

    def fail_after_mutation() -> None:
        skrl.config.jax.backend = "mutated"
        raise RuntimeError("failed")

    namespace["main"].__globals__["_main"] = fail_after_mutation
    with pytest.raises(RuntimeError, match="failed"):
        namespace["main"]()

    assert not hasattr(skrl.config.jax, "backend")
