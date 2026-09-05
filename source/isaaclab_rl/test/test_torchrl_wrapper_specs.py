# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free checks for :class:`IsaacLabTorchRLWrapper`'s spec conversion and step/reset contract.

The wrapper is built for real around a fake environment that mimics the parts of the Isaac Lab contract it
depends on (same-step auto-reset, in-place reuse of the reward/termination buffers, scalar episode logs, terminal
observations under ``extras["final_obs"]``) and is driven through TorchRL's public ``EnvBase`` API.
"""

import sys
import types

import gymnasium as gym
import pytest
import torch

pytest.importorskip("torchrl")

from tensordict import TensorDict  # noqa: E402
from torchrl.data import Bounded, Categorical, Composite, MultiCategorical, Unbounded  # noqa: E402
from torchrl.envs import ExplorationType, StepCounter, TransformedEnv, set_exploration_type  # noqa: E402
from torchrl.envs.utils import check_env_specs  # noqa: E402

from isaaclab_rl.torchrl import IsaacLabTorchRLWrapper, TorchRlPpoCfg, make_actor, train_ppo  # noqa: E402

NUM_ENVS = 4
CLOCK_SCALE = 100.0
"""Observations encode the per-environment step clock divided by this, keeping them inside the policy bounds."""


class _FakeEnvBase:
    """Stand-in for an Isaac Lab environment base class (registered on ``isaaclab.envs`` per test)."""


def _install_fake_env_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Points the env base classes the wrapper checks against at :class:`_FakeEnvBase` for one test.

    The classes are written straight into the module dictionary so that Isaac Lab's lazy module loader does
    not import the real environment stack; when Isaac Lab is not installed a stub module stands in for it.
    """
    try:
        import isaaclab.envs as module
    except ImportError:
        module = types.ModuleType("isaaclab.envs")
        monkeypatch.setitem(sys.modules, "isaaclab.envs", module)
    for name in ("DirectRLEnv", "ManagerBasedRLEnv"):
        monkeypatch.setitem(module.__dict__, name, _FakeEnvBase)


class _FakeUnwrapped(_FakeEnvBase):
    """Fake Isaac Lab environment.

    Each environment keeps a step clock that is encoded in the observations, reset to zero when the environment
    resets, and used to schedule time-outs. Rewards and done flags live in persistent buffers mutated in place.
    """

    def __init__(self, action_space=None, truncate_at=None, compute_final_obs=True, is_finite_horizon=False):
        self.num_envs = NUM_ENVS
        self.device = "cpu"
        self.cfg = types.SimpleNamespace(compute_final_obs=compute_final_obs, is_finite_horizon=is_finite_horizon)
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype="float32"),
                "critic": gym.spaces.Dict(
                    {"privileged": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(12,), dtype="float32")}
                ),
            }
        )
        self.single_action_space = action_space or gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype="float32")
        self.truncate_at = truncate_at or {}
        self.extras = {}
        self.reward_buf = torch.zeros(NUM_ENVS)
        self.reset_terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.clock = torch.zeros(NUM_ENVS)
        self.obs_buf = self._compute_obs()
        self.step_count = 0
        self.reset_calls = []
        self.last_actions = None
        self.last_seed = None
        self.closed = False

    def _compute_obs(self):
        policy = (self.clock / CLOCK_SCALE).unsqueeze(-1).expand(NUM_ENVS, 8).clone()
        privileged = self.clock.unsqueeze(-1).expand(NUM_ENVS, 12).clone()
        return {"policy": policy, "critic": {"privileged": privileged}}

    def seed(self, seed=-1):
        self.last_seed = seed
        return seed

    def reset(self, seed=None, options=None):
        self.reset_calls.append((seed, options))
        self.clock[:] = 0.0
        self.extras["log"] = {"Episode_Reward/track": torch.tensor(0.5)}
        self.obs_buf = self._compute_obs()
        return self.obs_buf, self.extras

    def step(self, actions):
        self.last_actions = actions
        self.step_count += 1
        self.clock += 1.0
        self.reward_buf[:] = float(self.step_count)
        self.reset_terminated[:] = False
        self.reset_time_outs[:] = False
        self.reset_time_outs[self.truncate_at.get(self.step_count, [])] = True
        done_ids = self.reset_time_outs.nonzero(as_tuple=True)[0]
        if len(done_ids) > 0:
            if self.cfg.compute_final_obs:
                self.extras["final_obs"] = self._compute_obs()
            self.clock[done_ids] = 0.0
            self.extras["log"] = {"Episode_Reward/track": torch.tensor(0.5)}
        self.obs_buf = self._compute_obs()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def close(self):
        self.closed = True


class _FakeEnv:
    """Outermost gymnasium-style wrapper around :class:`_FakeUnwrapped`."""

    def __init__(self, unwrapped):
        self.unwrapped = unwrapped

    def step(self, actions):
        return self.unwrapped.step(actions)

    def reset(self, *, seed=None, options=None):
        return self.unwrapped.reset(seed=seed, options=options)

    def close(self):
        self.unwrapped.close()


@pytest.fixture
def make_wrapper(monkeypatch):
    """Returns a factory building a real :class:`IsaacLabTorchRLWrapper` around a fake environment."""
    _install_fake_env_classes(monkeypatch)

    def _make(device=None, clip_actions=None, **fake_kwargs):
        fake = _FakeUnwrapped(**fake_kwargs)
        return IsaacLabTorchRLWrapper(_FakeEnv(fake), device=device, clip_actions=clip_actions), fake

    return _make


def _clock(td, *key):
    """Decodes the step clock from the policy observations stored under ``key``."""
    return (td[key if len(key) > 1 else key[0]][..., 0] * CLOCK_SCALE).tolist()


"""
Construction and spec conversion
"""


def test_rejects_non_isaaclab_env(make_wrapper):
    with pytest.raises(ValueError, match="must be inherited"):
        IsaacLabTorchRLWrapper(_FakeEnv(types.SimpleNamespace()))


def test_specs_mirror_isaaclab_spaces(make_wrapper):
    wrapper, _ = make_wrapper()

    assert isinstance(wrapper.observation_spec["policy"], Bounded)
    assert wrapper.observation_spec["policy"].shape == torch.Size([NUM_ENVS, 8])
    assert isinstance(wrapper.observation_spec["critic"], Composite)
    assert isinstance(wrapper.observation_spec["critic", "privileged"], Unbounded)
    assert wrapper.observation_spec["critic", "privileged"].shape == torch.Size([NUM_ENVS, 12])
    assert wrapper.reward_spec.shape == torch.Size([NUM_ENVS, 1])
    assert set(wrapper.done_keys) == {"done", "terminated", "truncated"}
    check_env_specs(wrapper)


@pytest.mark.parametrize(
    ("action_space", "clip_actions", "spec_type", "shape"),
    [
        (gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype="float32"), None, Bounded, (6,)),
        (gym.spaces.Box(low=-1.0, high=1.0, shape=(2, 3), dtype="float32"), None, Bounded, (2, 3)),
        (gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype="float32"), 2.0, Bounded, (6,)),
        (gym.spaces.Discrete(3), None, Categorical, (1,)),
        (gym.spaces.MultiDiscrete([3, 2]), None, MultiCategorical, (2,)),
    ],
    ids=["box", "box-2d", "box-clipped", "discrete", "multi-discrete"],
)
def test_action_spec_conversion(make_wrapper, action_space, clip_actions, spec_type, shape):
    wrapper, _ = make_wrapper(action_space=action_space, clip_actions=clip_actions)

    assert isinstance(wrapper.action_spec, spec_type)
    assert wrapper.action_spec.shape == torch.Size([NUM_ENVS, *shape])
    if clip_actions is not None:
        assert float(wrapper.action_spec.space.low.flatten()[0]) == -clip_actions
        assert float(wrapper.action_spec.space.high.flatten()[0]) == clip_actions
    if spec_type is not Bounded:
        assert wrapper.action_spec.dtype == torch.int64


@pytest.mark.parametrize(
    ("action_space", "clip_actions", "error"),
    [
        (gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(2))), None, NotImplementedError),
        (gym.spaces.Discrete(3), 1.0, ValueError),
    ],
    ids=["unsupported-space", "clip-on-discrete"],
)
def test_unsupported_action_configurations_raise(make_wrapper, action_space, clip_actions, error):
    with pytest.raises(error):
        make_wrapper(action_space=action_space, clip_actions=clip_actions)


"""
Step / reset contract
"""


def test_step_clips_actions_before_passing_to_env(make_wrapper):
    wrapper, fake = make_wrapper(clip_actions=1.0)
    td = wrapper.reset()
    td["action"] = torch.full((NUM_ENVS, 6), 5.0)

    wrapper.step(td)

    assert bool((fake.last_actions.abs() <= 1.0).all())


@pytest.mark.parametrize(
    ("is_finite_horizon", "expected_terminated", "expected_truncated"),
    [(False, False, True), (True, True, False)],
    ids=["infinite-horizon", "finite-horizon"],
)
def test_step_reports_time_outs(make_wrapper, is_finite_horizon, expected_terminated, expected_truncated):
    """Time-outs surface as ``truncated``, except in finite-horizon tasks where they are terminal."""
    wrapper, _ = make_wrapper(truncate_at={1: [0]}, is_finite_horizon=is_finite_horizon)
    td = wrapper.reset()
    td["action"] = torch.zeros(NUM_ENVS, 6)

    td = wrapper.step(td)

    for key in ("reward", "terminated", "truncated", "done"):
        assert td["next", key].shape == torch.Size([NUM_ENVS, 1])
    assert bool(td["next", "terminated"][0, 0]) is expected_terminated
    assert bool(td["next", "truncated"][0, 0]) is expected_truncated
    assert bool(td["next", "done"][0, 0]) is True
    assert not bool(td["next", "done"][1:].any())


def test_rollout_transitions_at_time_out(make_wrapper):
    """Per-step buffer copies, terminal observation on the done row, and post-reset continuation."""
    wrapper, fake = make_wrapper(truncate_at={2: [0]})

    rollout = wrapper.rollout(4, break_when_any_done=False)

    assert rollout["next", "reward"][0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert rollout["next", "truncated"][0, :, 0].tolist() == [False, True, False, False]
    assert rollout["next", "done"][0, :, 0].tolist() == [False, True, False, False]
    # env 0 timed out at step 2 with clock 2 and restarted from clock 0; env 1 ran through
    assert _clock(rollout[0], "next", "policy") == [1.0, 2.0, 1.0, 2.0]
    assert (rollout["next", "critic", "privileged"][0, 1] == 2.0).all()
    assert _clock(rollout[0], "policy") == [0.0, 1.0, 0.0, 1.0]
    assert _clock(rollout[1], "policy") == [0.0, 1.0, 2.0, 3.0]
    assert len(fake.reset_calls) == 1


def test_transformed_env_resets_transforms_on_done(make_wrapper):
    wrapper, fake = make_wrapper(truncate_at={2: [0]})
    env = TransformedEnv(wrapper, StepCounter())

    rollout = env.rollout(4, break_when_any_done=False)

    assert len(fake.reset_calls) == 1
    assert _clock(rollout[0], "next", "policy") == [1.0, 2.0, 1.0, 2.0]
    assert rollout["step_count"][0, :, 0].tolist() == [0, 1, 0, 1]
    assert rollout["step_count"][1, :, 0].tolist() == [0, 1, 2, 3]


def test_next_obs_at_time_out_is_nan_without_final_obs(make_wrapper):
    wrapper, _ = make_wrapper(truncate_at={2: [0]}, compute_final_obs=False)

    rollout = wrapper.rollout(4, break_when_any_done=False)

    assert bool(rollout["next", "policy"][0, 1].isnan().all())
    assert not bool(rollout["next", "policy"][0, [0, 2, 3]].isnan().any())
    assert not bool(rollout["next", "policy"][1:].isnan().any())


def test_reset_returns_initial_state_and_forwards_kwargs(make_wrapper):
    wrapper, fake = make_wrapper()

    td = wrapper.reset(seed=7, options={"key": "value"})

    assert fake.reset_calls == [(7, {"key": "value"})]
    for key in ("done", "terminated", "truncated"):
        assert not bool(td[key].any())
    assert td["policy"].shape == torch.Size([NUM_ENVS, 8])
    assert td["critic", "privileged"].shape == torch.Size([NUM_ENVS, 12])


def test_partial_reset_returns_current_observations_without_resetting(make_wrapper):
    wrapper, fake = make_wrapper()
    td = wrapper.rollout(3, break_when_any_done=False)[:, -1].exclude("next")
    td["_reset"] = torch.tensor([False, True, False, False]).unsqueeze(-1)

    td = wrapper.reset(td)

    # the masked env receives its current observation (clock 3); the others keep the last rollout step's (clock 2)
    assert len(fake.reset_calls) == 1
    assert _clock(td, "policy") == [2.0, 3.0, 2.0, 2.0]
    assert not bool(td["done"].any())


def test_reset_after_all_envs_done_does_not_reset_env(make_wrapper):
    wrapper, fake = make_wrapper(truncate_at={2: list(range(NUM_ENVS))})

    rollout = wrapper.rollout(4, break_when_any_done=False)

    # Isaac Lab already reset every environment inside step(); a second reset would re-randomize the new
    # episodes and overwrite the episode statistics just logged under extras["log"]
    assert len(fake.reset_calls) == 1
    assert _clock(rollout, "policy") == [[0.0, 1.0, 0.0, 1.0]] * NUM_ENVS


def test_seed_and_unwrapped_delegate_to_base_env(make_wrapper):
    wrapper, fake = make_wrapper()

    wrapper.set_seed(42)
    wrapper.set_seed(None)

    assert fake.last_seed == 42
    assert wrapper.unwrapped is fake


@pytest.mark.parametrize("transformed", [False, True], ids=["plain", "transformed"])
def test_close_delegates_to_wrapped_env(make_wrapper, transformed):
    wrapper, fake = make_wrapper()
    env = TransformedEnv(wrapper, StepCounter()) if transformed else wrapper
    env.rollout(2)

    env.close()

    assert fake.closed
    assert wrapper.is_closed is True


def test_episode_log_is_not_part_of_the_tensordict(make_wrapper):
    wrapper, fake = make_wrapper(truncate_at={1: [0]})

    td = wrapper.rollout(2, break_when_any_done=False)

    assert "info" not in td.keys()
    assert "Episode_Reward/track" in fake.extras["log"]


def test_tensordict_lives_on_requested_device(make_wrapper):
    wrapper, _ = make_wrapper(device="cpu")
    td = wrapper.reset()
    td["action"] = torch.zeros(NUM_ENVS, 6)

    td = wrapper.step(td)

    assert isinstance(td, TensorDict)
    assert td.device == torch.device("cpu")
    assert td["next", "policy"].device == torch.device("cpu")


"""
PPO example
"""


def test_train_ppo_learns_checkpoints_and_reloads(make_wrapper, tmp_path):
    wrapper, _ = make_wrapper(truncate_at={3: [0, 1]})
    cfg = TorchRlPpoCfg(
        seed=0,
        device="cpu",
        num_steps_per_env=4,
        max_iterations=2,
        save_interval=1,
        experiment_name="fake",
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
    )

    actor = train_ppo(wrapper, cfg, str(tmp_path))

    assert sorted(path.name for path in tmp_path.glob("model_*.pt")) == ["model_1.pt", "model_2.pt"]
    assert all(torch.isfinite(parameter).all() for parameter in actor.parameters())
    # the checkpoint holds the actor weights and reloads into a freshly built actor
    reloaded = make_actor(wrapper, cfg)
    reloaded.load_state_dict(torch.load(tmp_path / "model_2.pt"))
    td = wrapper.reset()
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        assert torch.equal(actor(td.clone())["action"], reloaded(td.clone())["action"])
