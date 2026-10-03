# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the RLinf extension module (isaaclab_contrib.rl.rlinf.extension).

These tests verify the pure-logic functions (config loading, obs/action conversion,
embodiment tag patching, and task registration) without requiring Isaac Sim or a GPU.
RLinf dependencies are mocked to keep the tests self-contained.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from enum import Enum
from unittest import mock

import numpy as np
import pytest
import torch
import yaml

# ---------------------------------------------------------------------------
# Mock the ``rlinf`` package so the extension module can be imported without
# having RLinf installed.  We create just enough structure for the top-level
# ``from rlinf.models.embodiment.gr00t import embodiment_tags`` to succeed.
# ---------------------------------------------------------------------------

_MOCK_EMBODIMENT_TAG_MAPPING: dict[str, int] = {
    "gr1": 24,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "new_embodiment": 31,
}


class _MockEmbodimentTag(Enum):
    GR1 = "gr1"
    OXE_DROID = "oxe_droid"
    AGIBOT_GENIE1 = "agibot_genie1"
    NEW_EMBODIMENT = "new_embodiment"


def _build_rlinf_mocks() -> dict[str, types.ModuleType]:
    """Build a dict of fake rlinf modules for ``sys.modules``."""
    mock_embodiment_tags = types.ModuleType("rlinf.models.embodiment.gr00t.embodiment_tags")
    mock_embodiment_tags.EmbodimentTag = _MockEmbodimentTag
    mock_embodiment_tags.EMBODIMENT_TAG_MAPPING = dict(_MOCK_EMBODIMENT_TAG_MAPPING)

    mock_simulation_io = types.ModuleType("rlinf.models.embodiment.gr00t.simulation_io")
    mock_simulation_io.OBS_CONVERSION = {}
    mock_simulation_io.ACTION_CONVERSION_N1D5 = {}
    mock_simulation_io.ACTION_CONVERSION_N1D7 = {}

    mock_isaaclab_env = types.ModuleType("rlinf.envs.isaaclab")
    mock_isaaclab_env.REGISTER_ISAACLAB_ENVS = {}

    mock_isaaclab_base = types.ModuleType("rlinf.envs.isaaclab.isaaclab_env")

    class _FakeIsaaclabBaseEnv:
        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
            self.isaaclab_env_id = ""
            self.cfg = cfg
            self.num_envs = num_envs
            self.device = "cpu"
            self.task_description = ""

    mock_isaaclab_base.IsaaclabBaseEnv = _FakeIsaaclabBaseEnv

    # Build the module hierarchy
    mods: dict[str, types.ModuleType] = {}
    for name in [
        "rlinf",
        "rlinf.models",
        "rlinf.models.embodiment",
        "rlinf.models.embodiment.gr00t",
        "rlinf.models.embodiment.gr00t.embodiment_tags",
        "rlinf.models.embodiment.gr00t.simulation_io",
        "rlinf.envs",
        "rlinf.envs.isaaclab",
        "rlinf.envs.isaaclab.isaaclab_env",
    ]:
        if name not in (
            "rlinf.models.embodiment.gr00t.embodiment_tags",
            "rlinf.models.embodiment.gr00t.simulation_io",
            "rlinf.envs.isaaclab",
            "rlinf.envs.isaaclab.isaaclab_env",
        ):
            mods[name] = types.ModuleType(name)
        else:
            pass  # added below

    mods["rlinf.models.embodiment.gr00t.embodiment_tags"] = mock_embodiment_tags
    mods["rlinf.models.embodiment.gr00t.simulation_io"] = mock_simulation_io
    mods["rlinf.envs.isaaclab"] = mock_isaaclab_env
    mods["rlinf.envs.isaaclab.isaaclab_env"] = mock_isaaclab_base

    # Wire sub-module attributes
    mods["rlinf.models.embodiment.gr00t"].embodiment_tags = mock_embodiment_tags
    mods["rlinf.models.embodiment.gr00t"].simulation_io = mock_simulation_io
    mods["rlinf.envs.isaaclab"].isaaclab_env = mock_isaaclab_base

    return mods


# Install mocks *before* importing the extension module
_rlinf_mocks = _build_rlinf_mocks()
sys.modules.update(_rlinf_mocks)

# Now we can safely import the extension
import isaaclab_contrib.rl.rlinf.extension as ext  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_YAML = {
    "env": {
        "train": {
            "init_params": {"id": "Isaac-Test-Task-v0"},
            "isaaclab": {
                "task_description": "Pick up the red cube",
                "main_images": "front_camera",
                "extra_view_images": ["left_wrist_camera", "right_wrist_camera"],
                "obs_converter_type": "isaaclab",
                "embodiment_tag": "test_robot",
                "embodiment_tag_id": 29,
                "states": [
                    {"key": "joint_pos", "slice": [0, 7]},
                    {"key": "joint_vel"},
                ],
                "gr00t_mapping": {
                    "video": {
                        "main_images": "video.room_view",
                        "extra_view_images": ["video.left_wrist", "video.right_wrist"],
                    },
                    "state": [
                        {"gr00t_key": "state.arm", "slice": [0, 7]},
                        {"gr00t_key": "state.hand", "slice": [7, 14]},
                    ],
                },
                "action_mapping": {
                    "prefix_pad": 3,
                    "suffix_pad": 2,
                },
            },
        },
        "eval": {
            "init_params": {"id": "Isaac-Test-Task-Eval-v0"},
            "isaaclab": {
                "task_description": "Pick up the red cube",
            },
        },
    }
}


@pytest.fixture(autouse=True)
def _reset_extension_state():
    """Reset extension module-level caches before each test."""
    ext._registered = False
    ext._full_cfg_cache = None
    # Reset mock registries
    sim_io = _rlinf_mocks["rlinf.models.embodiment.gr00t.simulation_io"]
    sim_io.OBS_CONVERSION.clear()
    sim_io.ACTION_CONVERSION_N1D5.clear()
    sim_io.ACTION_CONVERSION_N1D7.clear()
    _rlinf_mocks["rlinf.envs.isaaclab"].REGISTER_ISAACLAB_ENVS = {}
    _rlinf_mocks["rlinf.models.embodiment.gr00t.embodiment_tags"].EMBODIMENT_TAG_MAPPING = dict(
        _MOCK_EMBODIMENT_TAG_MAPPING
    )
    _rlinf_mocks["rlinf.models.embodiment.gr00t.embodiment_tags"].EmbodimentTag = _MockEmbodimentTag
    yield


@pytest.fixture()
def yaml_config_file() -> str:
    """Write the sample YAML config to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(_SAMPLE_YAML, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture()
def set_config_env(yaml_config_file: str):
    """Set ``RLINF_CONFIG_FILE`` for a single test."""
    with mock.patch.dict(os.environ, {"RLINF_CONFIG_FILE": yaml_config_file}):
        yield


# ---------------------------------------------------------------------------
# Tests: config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Tests for ``_load_full_cfg`` and ``_get_isaaclab_cfg``."""

    def test_load_full_cfg_success(self, set_config_env) -> None:
        """Config should load correctly from a valid YAML file."""
        cfg = ext._load_full_cfg()
        assert "env" in cfg
        assert cfg["env"]["train"]["init_params"]["id"] == "Isaac-Test-Task-v0"

    def test_load_full_cfg_caching(self, set_config_env) -> None:
        """Subsequent calls should return the same cached object."""
        cfg1 = ext._load_full_cfg()
        cfg2 = ext._load_full_cfg()
        assert cfg1 is cfg2

    def test_load_full_cfg_missing_env_var(self) -> None:
        """Should raise ValueError when RLINF_CONFIG_FILE is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RLINF_CONFIG_FILE", None)
            with pytest.raises(ValueError, match="RLINF_CONFIG_FILE not set"):
                ext._load_full_cfg()

    def test_get_isaaclab_cfg(self, set_config_env) -> None:
        """Should return the nested ``env.train.isaaclab`` section."""
        cfg = ext._get_isaaclab_cfg()
        assert cfg["task_description"] == "Pick up the red cube"
        assert cfg["main_images"] == "front_camera"
        assert cfg["obs_converter_type"] == "isaaclab"

    def test_get_isaaclab_cfg_missing_section(self, yaml_config_file: str) -> None:
        """Should return empty dict when the isaaclab section is absent."""
        minimal = {"env": {"train": {"init_params": {"id": "test"}}}}
        with open(yaml_config_file, "w") as f:
            yaml.dump(minimal, f)
        with mock.patch.dict(os.environ, {"RLINF_CONFIG_FILE": yaml_config_file}):
            cfg = ext._get_isaaclab_cfg()
            assert cfg == {}


# ---------------------------------------------------------------------------
# Tests: obs conversion
# ---------------------------------------------------------------------------


class TestObsConversion:
    """Tests for ``_convert_isaaclab_obs_to_gr00t``."""

    def test_extra_view_images_conversion(self, set_config_env) -> None:
        """Extra view images should be split into separate GR00T video keys."""
        ext._load_full_cfg()

        B, N, H, W, C = 4, 2, 64, 64, 3
        env_obs = {
            "extra_view_images": torch.randn(B, N, H, W, C),
            "task_descriptions": ["test"] * B,
        }
        result = ext._convert_isaaclab_obs_to_gr00t(env_obs)
        assert "video.left_wrist" in result
        assert "video.right_wrist" in result
        assert result["video.left_wrist"].shape == (B, 1, H, W, C)

    @pytest.mark.parametrize("language_key", [None, "annotation.human.task_description"])
    def test_task_descriptions_passthrough(self, set_config_env, language_key) -> None:
        """Use the checkpoint's language key without adding duplicate annotations."""
        cfg = ext._get_isaaclab_cfg()
        if language_key is not None:
            cfg["gr00t_mapping"]["language"] = language_key
        expected_key = language_key or "annotation.human.action.task_description"
        descs = ["task1", "task2"]
        assert ext._convert_isaaclab_obs_to_gr00t({"task_descriptions": descs}) == {expected_key: descs}
        # An empty observation still carries an (empty) task description.
        assert ext._convert_isaaclab_obs_to_gr00t({}) == {expected_key: []}

    def test_obs_state_slicing_consistency(self, set_config_env) -> None:
        """State slices must match the original tensor content after conversion."""
        ext._load_full_cfg()

        batch_size, state_dim = 8, 20
        states = torch.arange(batch_size * state_dim, dtype=torch.float32).reshape(batch_size, state_dim)
        obs = {"states": states, "task_descriptions": ["t"] * batch_size}
        gr00t_obs = ext._convert_isaaclab_obs_to_gr00t(obs)

        # gr00t_mapping.state[0]: slice [0,7] → state.arm
        np.testing.assert_allclose(gr00t_obs["state.arm"], states[:, None, :7].numpy(), atol=1e-6)
        # gr00t_mapping.state[1]: slice [7,14] → state.hand
        np.testing.assert_allclose(gr00t_obs["state.hand"], states[:, None, 7:14].numpy(), atol=1e-6)

    def test_image_value_preservation(self, set_config_env) -> None:
        """Pixel values should survive the obs conversion without corruption."""
        ext._load_full_cfg()

        B, H, W, C = 2, 8, 8, 3
        img = torch.rand(B, H, W, C)
        obs = {"main_images": img, "task_descriptions": ["t"] * B}
        gr00t_obs = ext._convert_isaaclab_obs_to_gr00t(obs)

        np.testing.assert_allclose(
            gr00t_obs["video.room_view"][:, 0],
            img.cpu().numpy(),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Tests: action conversion
# ---------------------------------------------------------------------------


class TestActionConversion:
    """Tests for ``_convert_gr00t_to_isaaclab_action``."""

    def test_concatenation_and_padding(self, set_config_env) -> None:
        """Actions should be concatenated and padded with prefix/suffix zeros."""
        ext._load_full_cfg()

        B, T, D1, D2 = 2, 4, 3, 5
        action_chunk = {
            "arm": np.random.randn(B, T, D1),
            "hand": np.random.randn(B, T, D2),
        }
        result = ext._convert_gr00t_to_isaaclab_action(action_chunk, chunk_size=2)

        # prefix_pad=3, suffix_pad=2, so total D = 3 + D1+D2 + 2 = 3+8+2 = 13
        assert result.shape == (B, 2, D1 + D2 + 3 + 2)
        # Check prefix padding is zeros
        np.testing.assert_array_equal(result[:, :, :3], 0.0)
        # Check suffix padding is zeros
        np.testing.assert_array_equal(result[:, :, -2:], 0.0)

    @pytest.mark.parametrize("keys", [None, ["arm", "hand"]])
    def test_no_padding(self, set_config_env, keys) -> None:
        """Concatenate in insertion order by default, or the declared environment action order."""
        ext._get_isaaclab_cfg()["action_mapping"] = {} if keys is None else {"keys": keys}
        action_chunk = {
            "hand": np.full((2, 3, 1), 2.0),
            "arm": np.full((2, 3, 2), 1.0),
        }
        result = ext._convert_gr00t_to_isaaclab_action(action_chunk, chunk_size=1)
        expected = [2.0, 1.0, 1.0] if keys is None else [1.0, 1.0, 2.0]
        np.testing.assert_array_equal(result, np.broadcast_to(expected, (2, 1, 3)))


# ---------------------------------------------------------------------------
# Tests: embodiment tag patching
# ---------------------------------------------------------------------------


class TestEmbodimentTagPatching:
    """Tests for ``_patch_embodiment_tags``."""

    def test_new_tag_added(self) -> None:
        """A custom tag not in the registry should be added."""
        cfg = {"embodiment_tag": "my_custom_robot", "embodiment_tag_id": 42}
        ext._patch_embodiment_tags(cfg)

        mapping = _rlinf_mocks["rlinf.models.embodiment.gr00t.embodiment_tags"].EMBODIMENT_TAG_MAPPING
        assert "my_custom_robot" in mapping
        assert mapping["my_custom_robot"] == 42

    def test_existing_tag_skipped(self) -> None:
        """A tag already in the registry should not be overwritten."""
        cfg = {"embodiment_tag": "gr1", "embodiment_tag_id": 99}
        ext._patch_embodiment_tags(cfg)

        mapping = _rlinf_mocks["rlinf.models.embodiment.gr00t.embodiment_tags"].EMBODIMENT_TAG_MAPPING
        # Should keep original value 24, not 99
        assert mapping["gr1"] == 24

    def test_default_tag_values(self) -> None:
        """Defaults should be ``new_embodiment`` with ID 31."""
        mapping = _rlinf_mocks["rlinf.models.embodiment.gr00t.embodiment_tags"].EMBODIMENT_TAG_MAPPING
        # Remove the default tag so the patch must add it from the defaults.
        del mapping["new_embodiment"]
        ext._patch_embodiment_tags({})

        assert mapping["new_embodiment"] == 31


# ---------------------------------------------------------------------------
# Tests: task registration
# ---------------------------------------------------------------------------


class TestTaskRegistration:
    """Tests for ``_register_isaaclab_envs``."""

    def test_tasks_registered_from_yaml(self, set_config_env) -> None:
        """Task IDs from train and eval sections should be registered."""
        ext._load_full_cfg()
        ext._register_isaaclab_envs()

        registry = _rlinf_mocks["rlinf.envs.isaaclab"].REGISTER_ISAACLAB_ENVS
        assert "Isaac-Test-Task-v0" in registry
        assert "Isaac-Test-Task-Eval-v0" in registry
        base = _rlinf_mocks["rlinf.envs.isaaclab.isaaclab_env"].IsaaclabBaseEnv
        for env_cls in registry.values():
            assert issubclass(env_cls, base)

    def test_no_duplicate_registration(self, set_config_env) -> None:
        """Calling register twice should not duplicate entries."""
        ext._load_full_cfg()
        ext._register_isaaclab_envs()
        ext._register_isaaclab_envs()

        registry = _rlinf_mocks["rlinf.envs.isaaclab"].REGISTER_ISAACLAB_ENVS
        assert len(registry) == 2

    def test_empty_config_no_registration(self, yaml_config_file: str) -> None:
        """Should warn and register nothing when no task IDs are found."""
        empty_cfg = {"env": {"train": {}, "eval": {}}}
        with open(yaml_config_file, "w") as f:
            yaml.dump(empty_cfg, f)
        with mock.patch.dict(os.environ, {"RLINF_CONFIG_FILE": yaml_config_file}):
            ext._register_isaaclab_envs()

        registry = _rlinf_mocks["rlinf.envs.isaaclab"].REGISTER_ISAACLAB_ENVS
        assert len(registry) == 0


# ---------------------------------------------------------------------------
# Tests: action chunks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hold_pose,ignore_terminations", [(False, False), (True, False), (True, True)])
def test_midchunk_reset_does_not_replay_old_episode_actions(monkeypatch, hold_pose, ignore_terminations):
    """Hold terminated/truncated worlds, preserve live actions, and release holds at the chunk boundary."""
    monkeypatch.setattr(
        ext, "_full_cfg_cache", {"env": {"train": {"isaaclab": {"hold_pose_on_midchunk_reset": hold_pose}}}}
    )
    base = _rlinf_mocks["rlinf.envs.isaaclab.isaaclab_env"].IsaaclabBaseEnv
    states = torch.tensor([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    applied = []

    def step(self, actions, auto_reset=True):
        applied.append(actions.clone())
        terminated = torch.tensor([len(applied) == 1, False, False])
        truncated = torch.tensor([False, len(applied) == 1, False])
        rewards = torch.ones(3)
        self._record_metrics(rewards, terminated, {})
        if ignore_terminations:
            terminated.zero_()  # RLinf clears these after recording metrics.
        return {"states": states}, rewards, terminated, truncated, {}

    def chunk_step(self, actions):
        return [self.step(action, auto_reset=False) for action in actions.unbind(dim=1)]

    monkeypatch.setattr(base, "step", step, raising=False)
    monkeypatch.setattr(base, "chunk_step", chunk_step, raising=False)
    env = ext._create_generic_env_wrapper("test")(None, 3, 0, 1, None)
    env.returns, env.success_once, env.elapsed_steps = torch.zeros(3), torch.zeros(3, dtype=torch.bool), torch.ones(3)
    actions = torch.arange(18, dtype=torch.bfloat16).reshape(3, 3, 2)
    original = actions.clone()
    env.chunk_step(actions)
    torch.testing.assert_close(actions, original)
    torch.testing.assert_close(applied[0], actions[:, 0])
    for index in (1, 2):
        expected = actions[:, index].clone()
        if hold_pose:
            expected[:2] = states[:2].to(expected)
        torch.testing.assert_close(applied[index], expected)
    env.step(actions[:, 0])
    torch.testing.assert_close(applied[-1], actions[:, 0])
    env.chunk_step(actions[:, :1])
    torch.testing.assert_close(applied[-1], actions[:, 0])


# ---------------------------------------------------------------------------
# Tests: converter registration
# ---------------------------------------------------------------------------


class TestConverterRegistration:
    """Tests for ``_register_gr00t_converters``."""

    @pytest.mark.parametrize("model_type", ["gr00t", "gr00t_n1d7"])
    def test_converters_registered(self, set_config_env, monkeypatch, model_type) -> None:
        """Register converters for both models, patching only the custom N1.5 loader."""
        ext._load_full_cfg()["actor"] = {"model": {"model_type": model_type}}
        ext._get_isaaclab_cfg()["data_config_class"] = "custom:DataConfig"
        gr00t = _rlinf_mocks["rlinf.models.embodiment.gr00t"]
        native_loader = object()
        monkeypatch.setattr(gr00t, "get_model", native_loader, raising=False)
        sim_io = _rlinf_mocks["rlinf.models.embodiment.gr00t.simulation_io"]
        ext.register()

        assert (gr00t.get_model is native_loader) == (model_type == "gr00t_n1d7")
        assert sim_io.OBS_CONVERSION["isaaclab"] is ext._convert_isaaclab_obs_to_gr00t
        for registry in (sim_io.ACTION_CONVERSION_N1D5, sim_io.ACTION_CONVERSION_N1D7):
            assert registry["isaaclab"] is ext._convert_gr00t_to_isaaclab_action

    def test_no_duplicate_converter_registration(self) -> None:
        """Should not overwrite existing converter entries."""
        sim_io = _rlinf_mocks["rlinf.models.embodiment.gr00t.simulation_io"]
        sentinel = lambda x: None  # noqa: E731
        registries = (sim_io.OBS_CONVERSION, sim_io.ACTION_CONVERSION_N1D5, sim_io.ACTION_CONVERSION_N1D7)
        for registry in registries:
            registry["isaaclab"] = sentinel

        ext._register_gr00t_converters({"obs_converter_type": "isaaclab"})

        for registry in registries:
            assert registry["isaaclab"] is sentinel
