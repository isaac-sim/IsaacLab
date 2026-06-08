# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for LEAPP observation-term input metadata."""

from types import SimpleNamespace

import torch

import isaaclab.utils.leapp.export_annotator as export_annotator
from isaaclab.utils.leapp import (
    ExportPatcher,
    XYZ_ELEMENT_NAMES,
    leapp_observation_input,
    resolve_leapp_element_names,
    resolve_leapp_observation_input_semantics,
)


def test_leapp_observation_input_marks_function():
    """Observation input metadata can be attached to function-based terms."""

    @leapp_observation_input(kind="state/body/position", element_names=[XYZ_ELEMENT_NAMES])
    def observation_term(env):
        return None

    semantics = resolve_leapp_observation_input_semantics(observation_term)

    assert semantics is not None
    assert semantics.kind == "state/body/position"
    assert semantics.element_names == [XYZ_ELEMENT_NAMES]
    assert resolve_leapp_element_names(semantics, observation_term) == [XYZ_ELEMENT_NAMES]


def test_leapp_observation_input_marks_class_instances():
    """Observation input metadata can be attached to class-based manager terms."""

    @leapp_observation_input(kind="state/gear/type")
    class ObservationTerm:
        def __call__(self, env):
            return None

    semantics_from_class = resolve_leapp_observation_input_semantics(ObservationTerm)
    semantics_from_instance = resolve_leapp_observation_input_semantics(ObservationTerm())

    assert semantics_from_class is not None
    assert semantics_from_class.kind == "state/gear/type"
    assert semantics_from_instance is semantics_from_class


def test_leapp_observation_input_supports_element_name_resolver():
    """Observation input metadata can resolve element names lazily."""

    def element_names_resolver(term):
        return [term.element_names]

    @leapp_observation_input(kind="state/custom", element_names_resolver=element_names_resolver)
    class ObservationTerm:
        element_names = ["a", "b", "c"]

        def __call__(self, env):
            return None

    term = ObservationTerm()
    semantics = resolve_leapp_observation_input_semantics(term)

    assert semantics is not None
    assert resolve_leapp_element_names(semantics, term) == [["a", "b", "c"]]


def test_leapp_observation_input_wrapper_annotates_processed_term(monkeypatch):
    """The export wrapper registers the configured observation value as the LEAPP input."""

    def add_offset(obs, offset: float):
        return obs + offset

    @leapp_observation_input(kind="state/body/position", element_names=[XYZ_ELEMENT_NAMES])
    def observation_term(env, multiplier: float):
        return env.value * multiplier

    real_env = SimpleNamespace(value=torch.tensor([[-1.0, 0.25, 2.0]]))
    term_cfg = SimpleNamespace(
        modifiers=[SimpleNamespace(func=add_offset, params={"offset": 1.0})],
        clip=(0.0, 2.0),
        scale=3.0,
    )
    semantics = resolve_leapp_observation_input_semantics(observation_term)
    patcher = ExportPatcher(export_method="onnx")
    patcher.task_name = "TestTask-v0"
    captured = {}

    def fake_input_tensors(task_name, tensor_semantics):
        captured["task_name"] = task_name
        captured["tensor_semantics"] = tensor_semantics
        return tensor_semantics.ref

    monkeypatch.setattr(export_annotator.annotate, "input_tensors", fake_input_tensors)

    wrapped = patcher._wrap_observation_input(
        observation_term,
        real_env,
        "policy",
        "shaft_pos",
        semantics,
        term_cfg,
    )
    returned = wrapped(SimpleNamespace(value=torch.full((1, 3), 100.0)), multiplier=2.0)

    expected = torch.tensor([[0.0, 4.5, 6.0]])
    torch.testing.assert_close(returned, expected)
    assert captured["task_name"] == "TestTask-v0"
    assert captured["tensor_semantics"].name == "shaft_pos"
    assert captured["tensor_semantics"].ref is returned
    assert captured["tensor_semantics"].kind == "state/body/position"
    assert captured["tensor_semantics"].element_names == [XYZ_ELEMENT_NAMES]
    assert captured["tensor_semantics"].extra == {"isaaclab_connection": "observation:policy:shaft_pos"}
