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


def test_cleared_composite_lets_the_generated_frame_stand(cpu_frames):
    # The mask is still built -- a backend may send it to the model as a guidance
    # signal -- so only the paste may be skipped, not the mask.
    cfg = make_cfg()
    cfg.cameras["cam_0"].composite_foreground = False
    runtime = VisualDRRuntime(cfg, num_envs=1, device="cpu")
    runtime.backend = Mock()
    runtime.backend.generate.side_effect = lambda sub, request: torch.full_like(sub.rgb, 99)
    runtime.activate()
    runtime.sync(make_env(1, [0], step=1))

    frame = make_frame(1, preserve=True)
    assert torch.equal(runtime.read("cam_0", frame.rgb, lambda: frame), torch.full_like(frame.rgb, 99))


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


def test_boundary_erosion_only_releases_selected_classes_at_the_outer_silhouette():
    segmentation = torch.zeros((1, 7, 8, 1), dtype=torch.int32)
    segmentation[:, 1:6, 2:7] = 1  # table
    segmentation[:, 1, 4] = 2  # thin robot structure on the outer edge
    segmentation[:, 3, 4] = 2  # robot touching the table interior
    labels = {"0": {"class": "ground"}, "1": {"class": "table"}, "2": {"class": "robot"}}
    cfg = CameraDRCfg(preserve_classes=("table", "robot"), boundary_erosion_px={"table": 1}, composite_foreground=False)
    actual = preserve_mask(segmentation, labels, cfg)
    expected = segmentation == 2
    expected[:, 2:5, 3:6] = True
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boundary_erosion_px": {"table": -1}},
        {"boundary_erosion_px": {"ground": 1}},
        {"boundary_erosion_px": {"table": 1}, "boundary_px": 1},
    ],
)
def test_boundary_erosion_rejects_invalid_or_conflicting_settings(kwargs):
    with pytest.raises(ValueError):
        CameraDRCfg(preserve_classes=("table",), **kwargs)


def test_union_erosion_shrinks_combined_mask_without_gaps_between_classes():
    segmentation = torch.zeros((1, 7, 8, 1), dtype=torch.int32)
    segmentation[:, 1:6, 1:4] = 1
    segmentation[:, 1:6, 4:7] = 2
    segmentation[:, 3, 3] = 9  # unknown but preserved
    labels = {"0": {"class": "ground"}, "1": {"class": "table"}, "2": {"class": "robot"}}
    cfg = CameraDRCfg(preserve_classes=("table", "robot"), boundary_erosion_px=1)
    expected = torch.zeros_like(segmentation, dtype=torch.bool)
    expected[:, 2:5, 2:6] = True
    assert torch.equal(preserve_mask(segmentation, labels, cfg), expected)


@pytest.mark.parametrize("radius,boundary", [(-1, 0), (1, 1)])
def test_union_erosion_rejects_negative_radius_and_dilation(radius, boundary):
    with pytest.raises(ValueError):
        CameraDRCfg(preserve_classes=("table",), boundary_erosion_px=radius, boundary_px=boundary)


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


def test_disabled_runtime_builds_from_a_config_that_names_no_backend():
    # The default config leaves `backend` MISSING. Constructing it used to raise,
    # which made "disabled" impossible to express with the shipped defaults.
    runtime = VisualDRRuntime(VisualDRCfg(), num_envs=2, device="cpu")
    assert runtime.enabled is False
    assert runtime.backend is None


def test_disabled_runtime_loads_no_model():
    cfg = make_cfg()
    cfg.enabled = False
    runtime = VisualDRRuntime(cfg, num_envs=1, device="cpu")
    # No backend exists to activate, so activate() cannot load anything.
    assert runtime.backend is None
    runtime.activate()
    runtime.offload()
    runtime.close()
    assert runtime.backend is None


def test_disabled_runtime_returns_the_frame_untouched(cpu_frames):
    cfg = make_cfg()
    cfg.enabled = False
    runtime = VisualDRRuntime(cfg, num_envs=1, device="cpu")
    runtime.activate()
    rgb = torch.full((1, 1, 2, 3), 7, dtype=torch.uint8)

    def explode() -> DRFrame:
        raise AssertionError("a disabled runtime must not fetch depth or segmentation")

    assert runtime.read("cam_0", rgb, explode) is rgb


def test_disabled_runtime_advances_no_scheduling_state(cpu_frames):
    cfg = make_cfg()
    cfg.enabled = False
    runtime = VisualDRRuntime(cfg, num_envs=2, device="cpu")
    runtime.sync(make_env(2, [0, 0], step=1))
    runtime.sync(make_env(2, [1, 1], step=2))
    assert int(runtime._episode_ids.sum()) == 0
    assert not bool(runtime._restyle.any())


def test_compositing_requires_classes_to_preserve():
    with pytest.raises(ValueError, match="must name the foreground"):
        CameraDRCfg()


def test_clearing_the_composite_makes_preserve_classes_unnecessary():
    cfg = CameraDRCfg(composite_foreground=False)
    assert cfg.preserve_classes == ()


def test_generated_frame_stands_as_is_without_the_composite(cpu_frames):
    cfg = make_cfg()
    cfg.cameras["cam_0"].composite_foreground = False
    runtime = VisualDRRuntime(cfg, num_envs=1, device="cpu")
    runtime.backend = Mock()
    runtime.backend.generate.side_effect = lambda sub, request: torch.full_like(sub.rgb, 99)
    runtime.activate()
    runtime.sync(make_env(1, [0], step=1))
    # The observation term hands over an empty mask in this mode, so nothing is
    # pasted back even though the frame carries a foreground.
    frame = DRFrame(
        torch.full((1, 1, 2, 3), 10, dtype=torch.uint8),
        torch.ones(1, 1, 2, 1),
        torch.zeros(1, 1, 2, 1, dtype=torch.bool),
    )
    assert torch.equal(runtime.read("cam_0", frame.rgb, lambda: frame), torch.full_like(frame.rgb, 99))


def test_mask_guidance_is_off_by_default():
    from isaaclab_contrib.visual_dr.cfg import CosmosBackendCfg, PromptBankCfg
    from isaaclab_contrib.visual_dr.cosmos import CosmosBackend

    cfg = CosmosBackendCfg(class_type=CosmosBackend, prompts=PromptBankCfg(variants=("a lab",)))
    # Stock cosmos-framework cannot accept a mask, so the default has to be off or
    # every default configuration would fail to load.
    assert cfg.mask_guidance is False


def test_mask_guidance_support_is_detected_not_assumed():
    from isaaclab_contrib.visual_dr.cosmos import _supports_mask_guidance

    class Stock:
        model_fields = {"prompt": None, "seed": None}

    class Patched:
        model_fields = {"prompt": None, "guided_generation_mask": None}

    class Args:
        def __init__(self, cls):
            self._cls = cls

        def get_sample_overrides_cls(self):
            return self._cls

    assert _supports_mask_guidance(Args(Stock)) is False
    assert _supports_mask_guidance(Args(Patched)) is True
    # An unexpected API shape is unsupported rather than an exception.
    assert _supports_mask_guidance(object()) is False


def test_a_checkpoint_path_that_does_not_exist_is_named_early(tmp_path):
    from isaaclab_contrib.visual_dr.cfg import CosmosBackendCfg, PromptBankCfg
    from isaaclab_contrib.visual_dr.cosmos import CosmosBackend

    cfg = CosmosBackendCfg(
        class_type=CosmosBackend,
        checkpoint=str(tmp_path / "Cosmos3-Nano-Transfer-Example"),
        prompts=PromptBankCfg(variants=("a lab",)),
    )
    backend = CosmosBackend(cfg)
    with pytest.raises(FileNotFoundError, match="looks like a path"):
        backend._build()


@pytest.mark.parametrize("checkpoint", ["Cosmos3-Nano", "nvidia/Cosmos3-Nano"])
def test_a_registered_checkpoint_name_is_not_mistaken_for_a_path(monkeypatch, checkpoint):
    from isaaclab_contrib.visual_dr import cosmos
    from isaaclab_contrib.visual_dr.cfg import CosmosBackendCfg, PromptBankCfg

    reached_loader = Mock(side_effect=RuntimeError("reached loader"))
    monkeypatch.setattr(cosmos, "_install_patches", reached_loader)
    cfg = CosmosBackendCfg(
        class_type=cosmos.CosmosBackend, checkpoint=checkpoint, prompts=PromptBankCfg(variants=("a lab",))
    )
    with pytest.raises(RuntimeError, match="reached loader"):
        cosmos.CosmosBackend(cfg).activate()
    reached_loader.assert_called_once()


def test_the_mask_is_built_even_when_the_composite_is_off():
    # A backend may send the mask to the model as guidance while the runtime does
    # not paste it back. Gating construction on the composite handed that backend
    # an all-zero mask, silently disabling the guidance.
    segmentation = torch.tensor([[[[1], [2]]]], dtype=torch.int32)
    labels = {"1": {"class": "robot"}, "2": {"class": "ground"}}
    cfg = CameraDRCfg(preserve_classes=("robot",), composite_foreground=False)
    assert preserve_mask(segmentation, labels, cfg).flatten().tolist() == [True, False]


def test_recipe_retains_task_fields_and_remote_settings(tmp_path):
    from isaaclab_contrib.visual_dr.cfg import CosmosBackendCfg, PromptBankCfg
    from isaaclab_contrib.visual_dr.cosmos import CosmosBackend
    from isaaclab_contrib.visual_dr.recipes import apply_visual_dr_recipe, remote_cosmos_cfg

    cfg = make_cfg()
    cfg.backend = CosmosBackendCfg(class_type=CosmosBackend, prompts=PromptBankCfg(variants=("lab",)))
    path = tmp_path / "recipe.yaml"
    path.write_text(
        "backend: {num_steps: 4, shift: 5, mask_guidance: true, mask_strength: 0.6}\n"
        "camera: {composite_foreground: false, boundary_erosion_px: 8}\n"
    )
    updated = apply_visual_dr_recipe(cfg, path)
    remote = remote_cosmos_cfg(updated.backend, (1, 2))
    assert cfg.backend.num_steps == 50
    assert remote.num_steps == 4 and remote.shift == 5
    assert remote.mask_guidance and remote.mask_strength == 0.6
    assert remote.mask_step_threshold is None
    assert remote.prompts.variants == ("lab",)
    assert remote.max_batch == 2
    assert updated.cameras["cam_0"].preserve_classes == ("robot",)
    assert not updated.cameras["cam_0"].composite_foreground
    assert updated.cameras["cam_0"].boundary_erosion_px == 8
    path.write_text("backend: {misspelled_setting: true}")
    with pytest.raises(TypeError, match="misspelled_setting"):
        apply_visual_dr_recipe(cfg, path)


def test_camera_sized_bucket_is_restored_after_failure():
    pytest.importorskip("cosmos_framework")
    from cosmos_framework.data.generator.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO

    from isaaclab_contrib.visual_dr.cfg import CosmosBackendCfg, PromptBankCfg
    from isaaclab_contrib.visual_dr.cosmos import CosmosBackend

    cfg = CosmosBackendCfg(
        class_type=CosmosBackend,
        prompts=PromptBankCfg(variants=("lab",)),
        resolution="480",
        aspect_ratio="4,3",
        native_resolution=True,
    )
    backend = CosmosBackend(cfg)
    original = [table["480"]["4,3"] for table in (IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO)]
    with pytest.raises(RuntimeError, match="generation failed"):
        with backend._resolution_override(640, 480):
            assert IMAGE_RES_SIZE_INFO["480"]["4,3"] == (640, 480)
            raise RuntimeError("generation failed")
    assert [table["480"]["4,3"] for table in (IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO)] == original


@pytest.mark.parametrize("kind", ["fixed", "unipc"])
@pytest.mark.parametrize("strength", [0.0, 0.6, 1.0])
def test_public_projection_preserves_source_or_disables_cleanly(kind, strength):
    pytest.importorskip("cosmos_framework")
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

    from isaaclab_contrib.visual_dr.cosmos_guidance import project_sampling

    sampler = (
        FixedStepSampler([1.0, 0.9375, 0.8333333333333, 0.625])
        if kind == "fixed"
        else UniPCSampler(tensor_kwargs={"device": "cpu"})
    )
    noise = torch.linspace(-2, 2, 19)
    source = torch.linspace(0, 1, 16)

    def velocity(state, timestep):
        return [x * 0.3 + x.sin() * 0.2 for x in state]

    unguided = sampler(velocity, [noise.clone()], num_steps=4, seed=[42])[0]
    actual = project_sampling(
        sampler,
        velocity,
        [noise.clone()],
        num_steps=4,
        seed=[42],
        source=source,
        mask=torch.ones_like(source) * strength,
        threshold=3,
        steps=4,
        train_steps=1000,
    )[0]
    # A prefix representing conditioning must remain untouched by the projection.
    torch.testing.assert_close(actual[:3], unguided[:3])
    if strength == 0:
        torch.testing.assert_close(actual, unguided)
    elif strength == 1:
        torch.testing.assert_close(actual[-16:], source)
    else:
        assert not torch.equal(actual[-16:], source)
        assert not torch.equal(actual, unguided)


@pytest.mark.parametrize("kind", ["fixed", "unipc"])
@pytest.mark.parametrize("threshold", [0, 3])
def test_public_adapter_matches_native_gitlab_hook(kind, threshold):
    import inspect

    pytest.importorskip("cosmos_framework")
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

    from isaaclab_contrib.visual_dr.cosmos_guidance import project_sampling

    sampler = (
        FixedStepSampler([1.0, 0.9375, 0.8333333333333, 0.625])
        if kind == "fixed"
        else UniPCSampler(tensor_kwargs={"device": "cpu"})
    )
    signature = sampler.__call__ if kind == "fixed" else sampler.forward
    if "state_projector" not in inspect.signature(signature).parameters:
        pytest.skip("Cross-implementation comparison requires the GitLab native hook")
    noise = torch.linspace(-2, 2, 19)
    source = torch.linspace(0, 1, 16)
    mask = torch.linspace(0, 1, 16)

    def velocity(state, timestep):
        return state * 0.3 + state.sin() * 0.2

    def native_projector(state, timestep, index):
        if index > threshold:
            return state
        result = state.clone()
        sigma = timestep.reshape(()) / 1000
        result[-16:] = mask * ((1 - sigma) * source + sigma * noise[-16:]) + (1 - mask) * state[-16:]
        return result

    expected = sampler(velocity, noise.clone(), num_steps=4, seed=42, state_projector=native_projector)
    actual = project_sampling(
        sampler,
        velocity,
        noise.clone(),
        num_steps=4,
        seed=42,
        source=source,
        mask=mask,
        threshold=threshold,
        steps=4,
        train_steps=1000,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_old_export_gets_modality_defaults_without_rewriting_checkpoint(tmp_path):
    import json

    from isaaclab_contrib.visual_dr.cosmos import _checkpoint_view

    source = tmp_path / "original"
    source.mkdir()
    config = {"model": {"config": {"diffusion_expert_config": {"enable_sound_modality_embedding": False}}}}
    original = json.dumps(config)
    (source / "config.json").write_text(original)
    (source / "weights.safetensors").write_bytes(b"weights")
    view = tmp_path / "view"
    view.mkdir()
    from pathlib import Path

    resolved = Path(_checkpoint_view(str(source), view))
    expert = json.loads((resolved / "config.json").read_text())["model"]["config"]["diffusion_expert_config"]
    assert not expert["enable_vision_modality_embeddings"]
    assert expert["enable_action_modality_embedding"]
    assert not expert["enable_sound_modality_embedding"]
    assert (source / "config.json").read_text() == original
    assert (resolved / "weights.safetensors").resolve() == source / "weights.safetensors"


def test_worker_startup_failure_survives_queue_deserialization():
    from isaaclab_contrib.visual_dr.remote import _Worker

    worker = object.__new__(_Worker)
    # Queue deserialization does not preserve Python string identity.
    worker._ready = Mock()
    worker._ready.get.return_value = ("".join(("fail", "ed")), "checkpoint load failed")
    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        worker.await_ready(1)
