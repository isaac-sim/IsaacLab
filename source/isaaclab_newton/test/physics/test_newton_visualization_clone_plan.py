# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for ClonePlan-aware Newton visualization model assembly."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_newton.physics import newton_manager as newton_manager_module
from isaaclab_newton.physics.newton_manager import NewtonManager

from pxr import Usd, UsdGeom

from isaaclab.cloner import ClonePlan
from isaaclab.sim import SimulationContext

_LABEL_ATTRS = ("body_label", "articulation_label", "joint_label", "shape_label")
_LABEL_SUFFIXES = {
    "body_label": "Body",
    "articulation_label": "Articulation",
    "joint_label": "Joint",
    "shape_label": "Shape",
}


class _FakeModelBuilder:
    """Small ``ModelBuilder`` substitute that records source copies and rewritten labels."""

    instances: list["_FakeModelBuilder"] = []

    def __init__(self, up_axis=None):
        self.up_axis = up_axis
        self.body_label: list[str] = []
        self.articulation_label: list[str] = []
        self.joint_label: list[str] = []
        self.shape_label: list[str] = []
        self.geometry_sources: list[str] = []
        self.add_usd_roots: list[str | None] = []
        self.world_slices: list[list[tuple[int, int, int, int]]] = []
        self._current_world: int | None = None
        self.instances.append(self)

    @property
    def body_count(self) -> int:
        return len(self.body_label)

    @property
    def shape_count(self) -> int:
        return len(self.shape_label)

    @property
    def joint_count(self) -> int:
        return len(self.joint_label)

    @property
    def particle_count(self) -> int:
        return 0

    def begin_world(self) -> None:
        self._current_world = len(self.world_slices)
        self.world_slices.append([])

    def end_world(self) -> None:
        self._current_world = None

    def add_usd(self, stage, root_path=None, ignore_paths=None, schema_resolvers=None) -> None:
        del stage, ignore_paths, schema_resolvers
        self.add_usd_roots.append(root_path)
        if root_path is None:
            return

        label_start = len(self.body_label)
        geometry_start = len(self.geometry_sources)
        for attr in _LABEL_ATTRS:
            getattr(self, attr).append(f"{root_path}/{_LABEL_SUFFIXES[attr]}")
        self.geometry_sources.append(root_path)
        self._record_world_slice(label_start, len(self.body_label), geometry_start, len(self.geometry_sources))

    def add_builder(self, builder: "_FakeModelBuilder", xform=None) -> None:
        del xform
        label_start = len(self.body_label)
        geometry_start = len(self.geometry_sources)
        for attr in _LABEL_ATTRS:
            getattr(self, attr).extend(getattr(builder, attr))
        self.geometry_sources.extend(builder.geometry_sources)
        self._record_world_slice(label_start, len(self.body_label), geometry_start, len(self.geometry_sources))

    def labels_for_world(self, world_id: int, attr: str) -> list[str]:
        labels = getattr(self, attr)
        return [
            label
            for label_start, label_end, _, _ in self.world_slices[world_id]
            for label in labels[label_start:label_end]
        ]

    def geometry_sources_for_world(self, world_id: int) -> list[str]:
        return [
            source
            for _, _, geometry_start, geometry_end in self.world_slices[world_id]
            for source in self.geometry_sources[geometry_start:geometry_end]
        ]

    def _record_world_slice(self, label_start: int, label_end: int, geometry_start: int, geometry_end: int) -> None:
        if self._current_world is not None:
            self.world_slices[self._current_world].append((label_start, label_end, geometry_start, geometry_end))


def _define_xform(stage: Usd.Stage, path: str, translation: tuple[float, float, float] | None = None) -> None:
    xform = UsdGeom.Xform.Define(stage, path)
    if translation is not None:
        xform.AddTranslateOp().Set(translation)


@pytest.fixture
def clone_plan_visualization_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    _define_xform(stage, "/World")
    _define_xform(stage, "/World/envs")

    for env_id in range(3):
        _define_xform(stage, f"/World/envs/env_{env_id}", (float(env_id) * 3.0, 0.0, 0.0))
        _define_xform(stage, f"/World/envs/env_{env_id}/Object")

    _define_xform(stage, "/World/envs/env_0/Object/source_0_visual")
    _define_xform(stage, "/World/envs/env_1/Object/source_1_visual")
    assert not stage.GetPrimAtPath("/World/envs/env_2/Object/source_0_visual").IsValid()
    return stage


def _build_with_clone_plan(stage: Usd.Stage, clone_plan: ClonePlan, monkeypatch: pytest.MonkeyPatch):
    _FakeModelBuilder.instances = []
    monkeypatch.setattr(newton_manager_module, "ModelBuilder", _FakeModelBuilder)
    monkeypatch.setattr(newton_manager_module, "SchemaResolverNewton", lambda: object())
    monkeypatch.setattr(newton_manager_module, "SchemaResolverPhysx", lambda: object())
    monkeypatch.setattr(
        SimulationContext,
        "instance",
        staticmethod(lambda: SimpleNamespace(get_clone_plan=lambda: clone_plan)),
    )
    monkeypatch.setattr(NewtonManager, "_num_envs", None)
    return NewtonManager._build_visualization_model_from_stage(stage)


def test_visualization_builder_uses_clone_plan_sources_and_rewrites_destination_labels(
    clone_plan_visualization_stage: Usd.Stage,
    monkeypatch: pytest.MonkeyPatch,
):
    """The PhysX-backed shadow model must mirror heterogeneous ClonePlan choices.

    Env 2 intentionally has no destination child geometry. The visualization
    model should still copy source env 0's geometry for env 2 while rewriting
    copied labels to env 2 paths so scene-data pose sync can find them.
    """
    clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor(
            [
                [True, False, True],
                [False, True, False],
            ],
            dtype=torch.bool,
        ),
    )

    builder = _build_with_clone_plan(clone_plan_visualization_stage, clone_plan, monkeypatch)

    assert isinstance(builder, _FakeModelBuilder)
    assert NewtonManager.get_num_envs() == 3
    assert len(builder.world_slices) == 3
    assert builder.geometry_sources_for_world(0) == ["/World/envs/env_0/Object"]
    assert builder.geometry_sources_for_world(1) == ["/World/envs/env_1/Object"]
    assert builder.geometry_sources_for_world(2) == ["/World/envs/env_0/Object"]

    for attr in _LABEL_ATTRS:
        suffix = _LABEL_SUFFIXES[attr]
        assert builder.labels_for_world(0, attr) == [f"/World/envs/env_0/Object/{suffix}"]
        assert builder.labels_for_world(1, attr) == [f"/World/envs/env_1/Object/{suffix}"]
        assert builder.labels_for_world(2, attr) == [f"/World/envs/env_2/Object/{suffix}"]


def test_rewrite_visualization_label_prefix_handles_trailing_source_slash():
    """Trailing slashes on ClonePlan roots must not drop the separator after the destination."""
    assert (
        NewtonManager._rewrite_visualization_label_prefix(
            "/World/envs/env_0/Object/Body",
            "/World/envs/env_0/Object/",
            "/World/envs/env_2/Object/",
        )
        == "/World/envs/env_2/Object/Body"
    )


def test_visualization_builder_falls_back_when_clone_mask_source_row_is_missing(
    clone_plan_visualization_stage: Usd.Stage,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Malformed ClonePlan masks should warn and fall back instead of raising ``IndexError``."""
    clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, False, False]], dtype=torch.bool),
    )

    with caplog.at_level("WARNING", logger="isaaclab_newton.physics.newton_manager"):
        builder = _build_with_clone_plan(clone_plan_visualization_stage, clone_plan, monkeypatch)

    assert isinstance(builder, _FakeModelBuilder)
    assert builder.geometry_sources_for_world(0) == ["/World/envs/env_0/Object"]
    assert builder.geometry_sources_for_world(1) == ["/World/envs/env_1"]
    assert builder.geometry_sources_for_world(2) == ["/World/envs/env_2"]
    assert any("exceeds clone_mask row count" in record.message for record in caplog.records)
