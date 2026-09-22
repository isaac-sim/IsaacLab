# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless contract tests: scheduling, masks and compositing on CPU tensors.

These do not cover CUDA transport, rendering or model output; the CUDA guard is
bypassed where it would otherwise be the only thing under test.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab_contrib.visual_dr import CameraDRCfg, DRFrame, PassthroughBackend, VisualDRRuntime
from isaaclab_contrib.visual_dr.cfg import DRBackendCfg, VisualDRCfg
from isaaclab_contrib.visual_dr.observations import preserve_mask, semantic_id


def make_cfg(num_cameras: int = 1, **kwargs) -> VisualDRCfg:
    cameras = {f"cam_{i}": CameraDRCfg(preserve_classes=("robot",)) for i in range(num_cameras)}
    defaults = {"enabled": True, "probability": 1.0, "cameras": cameras}
    defaults.update(kwargs)
    return VisualDRCfg(backend=DRBackendCfg(class_type=PassthroughBackend), **defaults)


def make_env(num_envs: int, episode_lengths: list[int], step: int) -> SimpleNamespace:
    return SimpleNamespace(
        common_step_counter=step,
        episode_length_buf=torch.tensor(episode_lengths, dtype=torch.long),
        num_envs=num_envs,
    )


def make_frame(num_envs: int = 2, preserve: bool = True) -> DRFrame:
    return DRFrame(
        torch.full((num_envs, 1, 2, 3), 10, dtype=torch.uint8),
        torch.ones(num_envs, 1, 2, 1),
        torch.full((num_envs, 1, 2, 1), preserve, dtype=torch.bool),
    )


@pytest.fixture
def cpu_frames(monkeypatch):
    """Bypass only the CUDA device check, so the rest of the path runs on CPU."""
    monkeypatch.setattr(DRFrame, "validate", lambda self: None)


def test_partial_reset_advances_only_the_reset_environments(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(), num_envs=3, device="cpu")
    runtime.sync(make_env(3, [4, 5, 6], step=1))
    before = runtime._episode_ids.clone()
    # Only environment 1 was reset by the previous step.
    runtime.sync(make_env(3, [5, 0, 7], step=2))
    assert torch.equal(runtime._episode_ids - before, torch.tensor([0, 1, 0]))


def test_action_chunk_gating_consumes_one_frame_per_chunk(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(decision_period=4), num_envs=1, device="cpu")
    consumed = []
    for step in range(1, 10):
        # episode_length_buf 0 on the first step marks the start of an episode.
        runtime.sync(make_env(1, [0 if step == 1 else step - 1], step=step))
        consumed.append(bool(runtime._consumed[0]))
    # The episode's first frame, then one every four actions.
    assert consumed == [True, False, False, False, True, False, False, False, True]


def test_style_is_stable_within_an_episode_and_changes_across_them(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(style="per_episode"), num_envs=2, device="cpu")
    runtime.sync(make_env(2, [0, 0], step=1))
    first = runtime._style_seeds.clone()
    runtime.sync(make_env(2, [1, 1], step=2))
    assert torch.equal(runtime._style_seeds, first)
    runtime.sync(make_env(2, [2, 0], step=3))
    assert runtime._style_seeds[0] == first[0]
    assert runtime._style_seeds[1] != first[1]


def test_probability_zero_never_generates(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(probability=0.0), num_envs=4, device="cpu")
    runtime.activate()
    runtime.sync(make_env(4, [0, 0, 0, 0], step=1))
    backend = Mock(wraps=runtime.backend)
    runtime.backend = backend
    frame = make_frame(4)
    assert torch.equal(runtime.read("cam_0", frame.rgb, lambda: frame), frame.rgb)
    backend.generate.assert_not_called()


def test_foreground_is_preserved_exactly_and_repeat_reads_match(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(), num_envs=2, device="cpu")
    runtime.backend = Mock()
    runtime.backend.generate.side_effect = lambda sub, request: torch.full_like(sub.rgb, 99)
    runtime.activate()
    runtime.sync(make_env(2, [0, 0], step=1))

    frame = make_frame(2, preserve=True)
    first = runtime.read("cam_0", frame.rgb, lambda: frame)
    assert torch.equal(first, frame.rgb), "preserved pixels must survive generation"

    # A second read of the same observation is served from cache, not regenerated.
    second = runtime.read("cam_0", frame.rgb, lambda: frame)
    assert torch.equal(first, second)
    assert runtime.backend.generate.call_count == 1


def test_background_is_replaced_where_nothing_is_preserved(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(), num_envs=1, device="cpu")
    runtime.backend = Mock()
    runtime.backend.generate.side_effect = lambda sub, request: torch.full_like(sub.rgb, 99)
    runtime.activate()
    runtime.sync(make_env(1, [0], step=1))
    frame = make_frame(1, preserve=False)
    assert torch.equal(runtime.read("cam_0", frame.rgb, lambda: frame), torch.full_like(frame.rgb, 99))


def test_generation_failure_can_pass_through_with_a_recorded_reason(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(on_error="passthrough"), num_envs=1, device="cpu")
    runtime.backend = Mock()
    runtime.backend.generate.side_effect = RuntimeError("model exploded")
    runtime.activate()
    runtime.sync(make_env(1, [0], step=1))
    frame = make_frame(1)
    assert torch.equal(runtime.read("cam_0", frame.rgb, lambda: frame), frame.rgb)
    assert runtime.errors and "model exploded" in runtime.errors[0]


def test_generation_failure_raises_by_default(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(), num_envs=1, device="cpu")
    runtime.backend = Mock()
    runtime.backend.generate.side_effect = RuntimeError("model exploded")
    runtime.activate()
    runtime.sync(make_env(1, [0], step=1))
    frame = make_frame(1)
    with pytest.raises(RuntimeError, match="model exploded"):
        runtime.read("cam_0", frame.rgb, lambda: frame)


def test_offloaded_runtime_returns_raw_frames(cpu_frames):
    runtime = VisualDRRuntime(make_cfg(), num_envs=1, device="cpu")
    runtime.activate()
    runtime.sync(make_env(1, [0], step=1))
    runtime.offload()
    backend = Mock(wraps=runtime.backend)
    runtime.backend = backend
    frame = make_frame(1)
    assert torch.equal(runtime.read("cam_0", frame.rgb, lambda: frame), frame.rgb)
    backend.generate.assert_not_called()


def test_preserve_mask_keeps_named_classes_and_unknown_ids():
    segmentation = torch.tensor([[[[1], [2]], [[3], [1]]]], dtype=torch.int32)
    labels = {"1": {"class": "robot"}, "2": {"class": "ground"}}
    cfg = CameraDRCfg(preserve_classes=("robot",))
    mask = preserve_mask(segmentation, labels, cfg)
    # 1 is named, 2 is background, 3 is untagged and therefore protected.
    assert mask.flatten().tolist() == [True, False, True, True]


def test_preserve_mask_can_randomize_unknown_ids():
    segmentation = torch.tensor([[[[1], [3]]]], dtype=torch.int32)
    labels = {"1": {"class": "robot"}}
    cfg = CameraDRCfg(preserve_classes=("robot",), unknown_policy="randomize")
    assert preserve_mask(segmentation, labels, cfg).flatten().tolist() == [True, False]


def test_preserve_mask_rejects_configurations_that_would_erase_everything():
    segmentation = torch.tensor([[[[2]]]], dtype=torch.int32)
    labels = {"2": {"class": "ground"}}
    with pytest.raises(ValueError, match="would be regenerated"):
        preserve_mask(segmentation, labels, CameraDRCfg(preserve_classes=("robot",)))


def test_preserve_mask_rejects_colorized_segmentation():
    with pytest.raises(ValueError, match="uncolored"):
        preserve_mask(
            torch.zeros(1, 1, 1, 1, dtype=torch.uint8),
            {"1": {"class": "robot"}},
            CameraDRCfg(preserve_classes=("robot",)),
        )


def test_boundary_dilation_grows_the_preserved_region():
    segmentation = torch.tensor([[[[1], [2], [2]]]], dtype=torch.int32)
    labels = {"1": {"class": "robot"}, "2": {"class": "ground"}}
    tight = preserve_mask(segmentation, labels, CameraDRCfg(preserve_classes=("robot",)))
    grown = preserve_mask(segmentation, labels, CameraDRCfg(preserve_classes=("robot",), boundary_px=1))
    assert tight.flatten().tolist() == [True, False, False]
    assert grown.flatten().tolist() == [True, True, False]


def test_semantic_ids_decode_rgba_keys_the_renderer_reports():
    # Observed from the RTX renderer with colorize disabled: the buffer holds
    # uncolored signed int32 IDs while idToLabels keys stay RGBA strings.
    assert semantic_id("(33, 243, 3, 255)") == -16518367
    assert semantic_id("(240, 4, 111, 255)") == -9501456
    assert semantic_id("(0, 0, 0, 0)") == 0
    assert semantic_id("7") == 7


def test_preserve_mask_accepts_rgba_label_keys():
    segmentation = torch.tensor([[[[-16518367], [0]]]], dtype=torch.int32)
    labels = {"(33, 243, 3, 255)": {"class": "cube_2"}, "(0, 0, 0, 0)": {"class": "BACKGROUND"}}
    mask = preserve_mask(segmentation, labels, CameraDRCfg(preserve_classes=("cube_2",)))
    assert mask.flatten().tolist() == [True, False]


def test_segmentation_control_map_recovers_the_renderer_palette():
    from isaaclab_contrib.visual_dr.cfg import CosmosBackendCfg, PromptBankCfg
    from isaaclab_contrib.visual_dr.cosmos import CosmosBackend

    cfg = CosmosBackendCfg(class_type=CosmosBackend, control_kind="seg", prompts=PromptBankCfg(variants=("a lab",)))
    backend = CosmosBackend.__new__(CosmosBackend)
    backend.cfg = cfg
    # (33, 243, 3, 255) packs little-endian to this signed id; the map must hand
    # back those very channels rather than an arbitrary hashed colour.
    frame = DRFrame(
        torch.zeros(1, 1, 1, 3, dtype=torch.uint8),
        torch.zeros(1, 1, 1, 1),
        torch.zeros(1, 1, 1, 1, dtype=torch.bool),
        torch.full((1, 1, 1, 1), -16518367, dtype=torch.int32),
    )
    control = backend._control_map(frame)
    assert control.shape == (1, 3, 1, 1)
    assert [round(float(c) * 255) for c in control.flatten()] == [33, 243, 3]


def test_segmentation_control_map_requires_the_buffer():
    from isaaclab_contrib.visual_dr.cfg import CosmosBackendCfg, PromptBankCfg
    from isaaclab_contrib.visual_dr.cosmos import CosmosBackend

    backend = CosmosBackend.__new__(CosmosBackend)
    backend.cfg = CosmosBackendCfg(
        class_type=CosmosBackend, control_kind="seg", prompts=PromptBankCfg(variants=("a lab",))
    )
    with pytest.raises(ValueError, match="needs the segmentation buffer"):
        backend._control_map(make_frame(1))


def test_frame_indexing_carries_segmentation():
    frame = DRFrame(
        torch.zeros(3, 1, 1, 3, dtype=torch.uint8),
        torch.zeros(3, 1, 1, 1),
        torch.zeros(3, 1, 1, 1, dtype=torch.bool),
        torch.arange(3, dtype=torch.int32).reshape(3, 1, 1, 1),
    )
    narrowed = frame.index(torch.tensor([2]))
    assert narrowed.segmentation is not None
    assert int(narrowed.segmentation.flatten()[0]) == 2


def _bank(**kwargs):
    from isaaclab_contrib.visual_dr.cfg import PromptBankCfg

    return PromptBankCfg(variants=("a room.", "another room."), **kwargs)


def make_progression_runtime(**bank_kwargs):
    cfg = make_cfg()
    cfg.backend.prompts = _bank(**bank_kwargs)
    return VisualDRRuntime(cfg, num_envs=1, device="cpu")


def test_prompt_progression_walks_once_across_the_configured_span():
    phrases = ("dawn.", "noon.", "dusk.", "midnight.")
    runtime = make_progression_runtime(progression=phrases, progression_steps=40)
    seen = []
    for step in range(1, 41):
        runtime._observation_index = step
        seen.append(runtime._prompt_for(0).rsplit(" ", 1)[-1])
    # Each phrase holds for a quarter of the span, in order.
    assert seen[0] == "dawn." and seen[-1] == "midnight."
    assert [seen[i] for i in (0, 10, 20, 30)] == list(phrases)


def test_prompt_progression_clamps_past_the_span_instead_of_wrapping():
    runtime = make_progression_runtime(progression=("dawn.", "midnight."), progression_steps=10)
    runtime._observation_index = 500
    assert runtime._prompt_for(0).endswith("midnight.")


def test_prompt_progression_keeps_the_episode_variant():
    runtime = make_progression_runtime(progression=("dusk.",), progression_steps=10)
    runtime._observation_index = 1
    assert runtime._prompt_for(0).startswith("a room.")
    assert runtime._prompt_for(1).startswith("another room.")


def test_prompts_are_untouched_without_a_progression():
    runtime = make_progression_runtime()
    runtime._observation_index = 7
    assert runtime._prompt_for(0) == "a room."


def _remote_cfg(devices, **kwargs):
    from isaaclab_contrib.visual_dr.cfg import PromptBankCfg, RemoteCosmosBackendCfg
    from isaaclab_contrib.visual_dr.remote import RemoteCosmosBackend

    return RemoteCosmosBackendCfg(
        class_type=RemoteCosmosBackend,
        devices=devices,
        prompts=PromptBankCfg(variants=("a lab",)),
        **kwargs,
    )


def test_remote_backend_requires_devices():
    with pytest.raises(ValueError, match="must name at least one GPU"):
        _remote_cfg(())


def test_remote_backend_rejects_duplicate_devices():
    with pytest.raises(ValueError, match="must be distinct"):
        _remote_cfg((1, 1))


def test_remote_backend_rejects_a_batch_larger_than_the_worker_count():
    # Each worker takes one frame per call, so the runtime has to chunk to fit.
    with pytest.raises(ValueError, match="exceeds 2 workers"):
        _remote_cfg((1, 2), max_batch=4)
