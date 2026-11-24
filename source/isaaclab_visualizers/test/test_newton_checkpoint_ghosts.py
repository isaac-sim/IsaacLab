# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint Ghosts slot layout and the tint the Newton GL panel writes to ghost shapes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import warp as wp
from isaaclab_visualizers.newton.newton_checkpoint_ghosts import NewtonCheckpointGhostPanel, display_opacity

from isaaclab.visualizers.checkpoint_ghost import CheckpointGhostCheckpoint, CheckpointGhostLayout

pytestmark = [pytest.mark.unit]

SLICE_SIZE = 2
POOL = 6
CHECKPOINT = CheckpointGhostCheckpoint(path="/runs/a/model_1.pt", run="a", name="model_1.pt", modified=0.0)


def _panel(layout: CheckpointGhostLayout) -> NewtonCheckpointGhostPanel:
    return NewtonCheckpointGhostPanel(SimpleNamespace(layout=layout))


def _model() -> SimpleNamespace:
    """Model with one shared static shape and one visible shape per world of three slots."""
    worlds = np.array([-1] + list(range(3 * SLICE_SIZE)))
    return SimpleNamespace(
        shape_world=wp.array(worlds, dtype=wp.int32, device="cpu"),
        shape_color=wp.array(np.full((len(worlds), 3), 0.5), dtype=wp.vec3, device="cpu"),
        shape_opacity=None,
    )


def test_only_slots_with_a_checkpoint_are_visible_and_limited_per_slot() -> None:
    layout = CheckpointGhostLayout(POOL, 3)
    layout.slots[0].checkpoint = CHECKPOINT
    layout.slots[2].checkpoint = CHECKPOINT

    # two loaded slots split the six envs into slices of three; the second loaded slot starts at env 3
    assert layout.visible_env_ids(1) == [0, 3]
    assert layout.visible_env_ids(99) == [0, 1, 2, 3, 4, 5]

    layout.slots[2].visible = False

    assert layout.visible_env_ids(2) == [0, 1]


def test_the_environments_are_split_evenly_among_the_loaded_slots() -> None:
    layout = CheckpointGhostLayout(32, 8)
    layout.slots[0].checkpoint = CHECKPOINT
    assert layout.slice_size == 32

    layout.slots[1].checkpoint = CHECKPOINT
    assert layout.slice_size == 16

    layout.slots[5].checkpoint = CHECKPOINT
    assert layout.slice_size == 10  # 32 // 3; two envs belong to no slot
    assert layout.slot_of_env(np.array([-1, 0, 9, 10, 29, 30, 31])).tolist() == [-1, 0, 0, 1, 5, -1, -1]

    layout.slots[1].checkpoint = None
    assert layout.slice_size == 16
    assert layout.slot_of_env(np.array([0, 15, 16, 31])).tolist() == [0, 0, 5, 5]


def test_tint_applies_to_active_ghost_worlds_only_and_follows_slot_changes() -> None:
    layout = CheckpointGhostLayout(POOL, 3)
    layout.slots[0].checkpoint = CHECKPOINT
    layout.slots[1].checkpoint = CHECKPOINT
    layout.slots[1].opacity = 0.4
    layout.slots[1].tint = 1.0
    panel, model = _panel(layout), _model()

    panel.apply_tint(model)

    colors, opacity = model.shape_color.numpy(), model.shape_opacity.numpy()
    ghost_shapes = [4, 5, 6]  # worlds 3 to 5: two loaded slots split the six envs into slices of three
    assert np.allclose(colors[ghost_shapes], layout.slots[1].color)
    assert np.allclose(opacity[ghost_shapes], display_opacity(0.4))
    untouched = [0, 1, 2, 3]
    assert np.allclose(colors[untouched], 0.5)
    assert np.allclose(opacity[untouched], 1.0)

    layout.slots[1].checkpoint = None
    panel.apply_tint(model)
    assert np.allclose(model.shape_color.numpy(), 0.5)
    assert np.allclose(model.shape_opacity.numpy(), 1.0)


def test_slot_changes_are_reported_once() -> None:
    layout = CheckpointGhostLayout(POOL, 3)
    panel = _panel(layout)
    assert panel.consume_slots_changed()
    assert not panel.consume_slots_changed()

    layout.slots[1].checkpoint = CHECKPOINT

    assert panel.consume_slots_changed()


def test_tint_strength_blends_between_native_and_ghost_color() -> None:
    layout = CheckpointGhostLayout(POOL, 3)
    layout.slots[0].checkpoint = CHECKPOINT
    layout.slots[1].checkpoint = CHECKPOINT
    layout.slots[1].tint = 0.5
    panel, model = _panel(layout), _model()

    panel.apply_tint(model)

    ghost_shape = 4
    expected = 0.5 * 0.5 + 0.5 * np.asarray(layout.slots[1].color)
    assert np.allclose(model.shape_color.numpy()[ghost_shape], expected)

    layout.slots[1].tint = 0.0
    panel.apply_tint(model)
    assert np.allclose(model.shape_color.numpy()[ghost_shape], 0.5)


def test_a_changed_ghost_color_is_written_even_when_nothing_else_changed() -> None:
    layout = CheckpointGhostLayout(POOL, 3)
    layout.slots[0].checkpoint = CHECKPOINT
    layout.slots[1].checkpoint = CHECKPOINT
    layout.slots[1].tint = 1.0
    panel, model = _panel(layout), _model()
    panel.apply_tint(model)

    layout.slots[1].color = (0.0, 0.0, 1.0)
    panel.apply_tint(model)

    assert np.allclose(model.shape_color.numpy()[4], (0.0, 0.0, 1.0))


def _viewer_with_batch() -> SimpleNamespace:
    """Viewer with one batch holding a textured, metallic instance for each of the three slots' first worlds."""
    worlds = np.array([0, SLICE_SIZE, 2 * SLICE_SIZE])
    native = np.tile([0.5, 0.7, 0.0, 1.0], (3, 1)).astype(np.float32)
    batch = SimpleNamespace(
        worlds=wp.array(worlds, dtype=wp.int32, device="cpu"),
        materials=wp.array(native, dtype=wp.vec4, device="cpu"),
    )
    return SimpleNamespace(_shape_instances={"batch": batch}, model_changed=False)


def test_a_tinted_ghost_loses_its_texture_and_fades_its_metallic_shading_with_the_tint() -> None:
    layout = CheckpointGhostLayout(POOL, 3)
    for slot in layout.slots:
        slot.checkpoint = CHECKPOINT
    layout.slots[1].tint = 1.0
    layout.slots[2].tint = 0.5
    panel, viewer = _panel(layout), _viewer_with_batch()

    panel.apply_materials(viewer)

    materials = viewer._shape_instances["batch"].materials.numpy()
    assert np.allclose(materials[1], [0.5, 0.0, 0.0, 0.0])
    assert np.allclose(materials[2], [0.5, 0.35, 0.0, 0.0])
    assert np.allclose(materials[0], [0.5, 0.7, 0.0, 1.0])
    assert viewer.model_changed

    layout.slots[1].tint = 0.0
    layout.slots[2].tint = 0.0
    panel.apply_materials(viewer)
    assert np.allclose(viewer._shape_instances["batch"].materials.numpy(), [0.5, 0.7, 0.0, 1.0])


def test_tinting_keeps_the_opacity_the_model_already_has() -> None:
    layout = CheckpointGhostLayout(POOL, 3)
    layout.slots[0].checkpoint = layout.slots[1].checkpoint = CHECKPOINT
    model = _model()
    base = np.full(len(model.shape_world.numpy()), 0.4, dtype=np.float32)
    model.shape_opacity = wp.array(base, dtype=wp.float32, device="cpu")
    panel = _panel(layout)

    panel.apply_tint(model)

    opacity = model.shape_opacity.numpy()
    ghost = model.shape_world.numpy() >= 3  # two loaded slots split the six envs into slices of three
    assert (opacity[~ghost & (model.shape_world.numpy() >= 0)] == 0.4).all() and opacity[0] == 0.4
    assert np.allclose(opacity[ghost], display_opacity(layout.slots[1].opacity))


class _TextImgui:
    """Imgui double that records the text, tooltips, and headers a section draws."""

    Col_ = SimpleNamespace(text=0)
    Cond_ = SimpleNamespace(appearing=0, first_use_ever=0)
    HoveredFlags_ = SimpleNamespace(allow_when_disabled=0)
    ColorEditFlags_ = SimpleNamespace(no_tooltip=0)

    def __init__(self, open_header: bool = True):
        self.open_header = open_header
        self.texts, self.tooltips, self.headers = [], [], []

    def set_next_item_open(self, *args):
        pass

    def collapsing_header(self, label):
        self.headers.append(label)
        return self.open_header

    def is_item_hovered(self, *args):
        return True

    def set_tooltip(self, text):
        self.tooltips.append(text)

    def ImVec4(self, *args):
        return args

    ImVec2 = ImVec4

    def text(self, text):
        self.texts.append(text)

    text_wrapped = text

    def text_colored(self, color, text):
        self.texts.append(text)

    def slider_float(self, label, value, *args):
        return False, value

    def checkbox(self, label, value):
        return False, value

    def tree_node(self, label):
        return False

    def get_frame_height(self):
        return 20

    def get_cursor_screen_pos(self):
        return SimpleNamespace(x=0, y=0)

    def get_window_draw_list(self):
        return SimpleNamespace(add_rect=lambda *args: None)

    def __getattr__(self, name):
        return lambda *args, **kwargs: False


def test_the_section_reports_the_simulated_and_drawn_environments_and_follows_the_checkpoints() -> None:
    layout = CheckpointGhostLayout(32, 8)
    layout.slots[0].checkpoint = layout.slots[1].checkpoint = CHECKPOINT
    panel = NewtonCheckpointGhostPanel(SimpleNamespace(layout=layout, slot_metrics=lambda index: None))
    owner = SimpleNamespace(visible_env_count=3)
    imgui = _TextImgui()

    panel.draw(imgui, owner=owner)

    assert "Showing 6/32 envs: 3 per slot x 2 ckpts" in imgui.texts

    layout.slots[2].checkpoint = CHECKPOINT
    imgui = _TextImgui()
    panel.draw(imgui, owner=owner)

    assert "Showing 9/32 envs: 3 per slot x 3 ckpts" in imgui.texts


def test_without_the_flag_the_section_is_greyed_out_and_explains_how_to_enable_it() -> None:
    from isaaclab_visualizers.newton.newton_checkpoint_ghosts import draw_checkpoint_ghost_offer

    imgui = _TextImgui()

    draw_checkpoint_ghost_offer(imgui)

    assert imgui.headers == ["Checkpoint Ghosts (not enabled)"]
    assert any("Checkpoint visualization" in tip for tip in imgui.tooltips)
    assert any("--checkpoint_ghosts" in text for text in imgui.texts)


def test_the_opacity_slider_eases_to_just_under_the_opaque_path() -> None:
    values = [display_opacity(s) for s in (0.1, 0.4, 0.8, 1.0)]

    assert values == sorted(values)
    assert display_opacity(0.4) == pytest.approx(0.64)
    assert display_opacity(1.0) < 0.999  # Newton re-batches shapes at 0.999 and above
    assert display_opacity(0.95) > 0.99


class _BrowserImgui(_TextImgui):
    """Imgui double for the checkpoint window: it clicks the entries it is told to."""

    InputTextFlags_ = SimpleNamespace(enter_returns_true=0)

    def __init__(self, clicks=()):
        super().__init__()
        self.clicks, self.drawn = list(clicks), []

    def begin(self, title, opened):
        return True, opened

    def button(self, label):
        return label in self.clicks

    def selectable(self, label, selected):  # imgui requires the selected flag, as the real binding does
        self.drawn.append(label)
        return label in self.clicks, False

    def input_text(self, label, text, flags):
        return False, text


class _Browser:
    """Controller whose filesystem is a dictionary of folders."""

    def __init__(self, layout, tree):
        self.layout, self.tree, self.listed = layout, tree, []
        self.checkpoint_root = "/runs/task"
        self.added = []

    other_tasks: dict = {}

    def other_task(self, directory):
        return self.other_tasks.get(directory)

    def browse(self, directory):
        self.listed.append(directory)
        if directory not in self.tree:
            raise FileNotFoundError(2, "No such file or directory")
        folders, files = self.tree[directory]
        return folders, [CheckpointGhostCheckpoint(f"{directory}/{n}", "r", n, 0.0) for n in files]

    def add_ghost(self, checkpoint):
        self.added.append(checkpoint.path)
        return


def test_the_checkpoint_window_navigates_folders_and_remembers_where_it_was():
    tree = {"/runs/task": (["run_a"], []), "/runs/task/run_a": ([], ["model_1.pt"]), "/runs": (["task", "skrl"], [])}
    controller = _Browser(CheckpointGhostLayout(POOL, 3), tree)
    panel = NewtonCheckpointGhostPanel(controller)

    panel._open_window()
    panel._draw_window(_BrowserImgui(clicks=["run_a/##folder"]))
    assert panel._directory == "/runs/task/run_a"

    panel._draw_window(_BrowserImgui(clicks=["Up"]))
    assert panel._directory == "/runs/task"
    panel._draw_window(_BrowserImgui(clicks=["Up"]))
    assert panel._directory == "/runs"  # it can leave the experiment folder

    panel._window_open = False  # closing the window keeps the folder
    panel._open_window()
    assert panel._directory == "/runs"
    assert controller.listed[-1] == "/runs"


def test_choosing_a_checkpoint_keeps_the_folder_and_an_unreadable_folder_is_reported():
    tree = {"/runs/task": ([], ["model_1.pt"])}
    controller = _Browser(CheckpointGhostLayout(POOL, 3), tree)
    panel = NewtonCheckpointGhostPanel(controller)
    panel._open_window()

    panel._draw_window(_BrowserImgui(clicks=["model_1.pt   1970-01-01 00:00##/runs/task/model_1.pt"]))

    assert controller.added == ["/runs/task/model_1.pt"] and not panel._window_open
    panel._open_window()
    assert panel._directory == "/runs/task"

    panel._go("/missing")
    assert "Cannot read this folder" in panel._browse_error


def test_runs_of_another_task_are_flagged_in_the_checkpoint_window():
    tree = {"/runs/task": (["osc", "same"], []), "/runs/task/osc": ([], ["model_1.pt"])}
    controller = _Browser(CheckpointGhostLayout(POOL, 3), tree)
    controller.other_tasks = {"/runs/task/osc": "Isaac-Reach-Franka-OSC"}
    panel = NewtonCheckpointGhostPanel(controller)
    panel._open_window()

    imgui = _BrowserImgui()
    panel._draw_window(imgui)

    assert "osc/   (Isaac-Reach-Franka-OSC)##folder" in imgui.drawn and "same/##folder" in imgui.drawn

    panel._go("/runs/task/osc")
    imgui = _BrowserImgui()
    panel._draw_window(imgui)

    assert any("Isaac-Reach-Franka-OSC, not this task" in text for text in imgui.texts)


class _DisablingImgui(_TextImgui):
    """Imgui double that counts how deeply disabled the controls it draws are."""

    def __init__(self):
        super().__init__()
        self.depth, self.deepest, self.disabled_buttons = 0, 0, []

    def begin_disabled(self):
        self.depth += 1
        self.deepest = max(self.deepest, self.depth)

    def end_disabled(self):
        self.depth -= 1

    def button(self, label):
        self.disabled_buttons.append((label, self.depth > 0))
        return False


def test_the_ghost_controls_are_disabled_in_the_rtx_view_with_a_note_to_switch_renderer() -> None:
    layout = CheckpointGhostLayout(POOL, 3)
    layout.slots[0].checkpoint = CHECKPOINT
    panel = _panel(layout)
    panel._window_open = True
    panel._draw_window = Mock()

    imgui = _DisablingImgui()
    panel.draw(imgui, rtx_view=True)

    assert any("RTX: switch the Renderer to OpenGL" in text for text in imgui.texts)
    assert imgui.depth == 0 and all(disabled for _, disabled in imgui.disabled_buttons) and imgui.disabled_buttons
    panel._draw_window.assert_not_called()

    imgui = _DisablingImgui()
    panel.draw(imgui, rtx_view=False)

    assert not any("RTX:" in text for text in imgui.texts)
    assert imgui.depth == 0 and not any(disabled for _, disabled in imgui.disabled_buttons)
    panel._draw_window.assert_called_once()
