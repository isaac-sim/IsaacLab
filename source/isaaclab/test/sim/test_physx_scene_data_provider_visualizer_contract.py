# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for PhysxSceneDataProvider visualizer-facing contracts."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from isaaclab_physx.scene_data_providers import PhysxSceneDataProvider

from isaaclab.cloner import ClonePlan

PROVIDER_MOD = "isaaclab_physx.scene_data_providers.physx_scene_data_provider"


def _silent_stage() -> SimpleNamespace:
    """Stage stub whose ``GetPrimAtPath`` returns an invalid prim — env xforms read as zero."""
    return SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: False))


@pytest.fixture
def stub_provider():
    """Bare :class:`PhysxSceneDataProvider` with all buffer attrs initialized to defaults.

    Tests assign ``_simulation_context`` and ``_stage`` themselves; everything else is the
    pre-build state the build path expects.
    """
    p = object.__new__(PhysxSceneDataProvider)
    p._device = "cpu"
    p._xform_views = {}
    p._view_body_index_map = {}
    p._view_order_tensors = {}
    p._pose_buf_num_bodies = 0
    p._positions_buf = None
    p._orientations_buf = None
    p._covered_buf = None
    p._xform_mask_buf = None
    return p


@pytest.fixture
def newton_stub(monkeypatch):
    """Stub the ``isaaclab_newton`` newton-prebuild module and the side-effect helpers.

    Returned :class:`SimpleNamespace` exposes:

    * ``calls`` — list of kwargs from each prebuild invocation,
    * ``model`` / ``state_obj`` — what prebuild returns; tests can override before invoking.
    """
    state = SimpleNamespace(
        calls=[],
        model=SimpleNamespace(body_label=[], articulation_label=[]),
        state_obj=object(),
    )

    def _prebuild(**kwargs):
        state.calls.append(dict(kwargs))
        return state.model, state.state_obj

    monkeypatch.setitem(
        sys.modules, "isaaclab_newton.cloner.newton_replicate", SimpleNamespace(newton_visualizer_prebuild=_prebuild)
    )
    monkeypatch.setattr(f"{PROVIDER_MOD}.UsdGeom.GetStageUpAxis", lambda stage: "Z")
    monkeypatch.setattr(f"{PROVIDER_MOD}.replace_newton_shape_colors", lambda m, s: None)
    return state


def test_get_newton_model_returns_model_when_sync_enabled(stub_provider):
    """Callers receive the full Newton model from :meth:`get_newton_model`."""
    stub_provider._needs_newton_sync = True
    stub_provider._newton_model = "full-model"
    assert stub_provider.get_newton_model() == "full-model"


def test_build_from_clone_plans_populates_provider_state(stub_provider, newton_stub):
    """Building from per-group clone plans sets model, state, and rigid-body paths.

    Asserts the provider derives its own (sources, destinations, mask) from the plans
    without consulting any auxiliary spec object: representative source paths are recovered
    from ``dest_template.format(<first env using each prototype>)``, masks are concatenated
    along the prototype axis, and per-env positions are read from stage xforms.
    """
    newton_stub.model = SimpleNamespace(
        body_label=["/World/envs/env_0/Object/A"],
        articulation_label=["/World/envs/env_0/Robot"],
    )
    plans = {
        "/World/envs/env_{}/Object": ClonePlan(
            dest_template="/World/envs/env_{}/Object",
            prototype_paths=["/World/template/Object/proto_0", "/World/template/Object/proto_1"],
            # proto 0 → env 0, 2 ; proto 1 → env 1, 3
            clone_mask=torch.tensor([[True, False, True, False], [False, True, False, True]], dtype=torch.bool),
        ),
        "/World/envs/env_{}/Robot": ClonePlan(
            dest_template="/World/envs/env_{}/Robot",
            prototype_paths=["/World/template/Robot/proto_0"],
            clone_mask=torch.ones((1, 4), dtype=torch.bool),
        ),
    }
    stub_provider._simulation_context = SimpleNamespace(get_clone_plans=lambda: plans)
    stub_provider._stage = _silent_stage()

    stub_provider._build_newton_model_from_clone_plans()

    assert stub_provider._newton_model is newton_stub.model
    assert stub_provider._newton_state is newton_stub.state_obj
    assert stub_provider._rigid_body_paths == newton_stub.model.body_label
    assert stub_provider._rigid_body_view_paths == newton_stub.model.body_label + newton_stub.model.articulation_label
    assert stub_provider._num_envs_at_last_newton_build == 4
    assert stub_provider._last_newton_model_build_source == "built"

    kw = newton_stub.calls[-1]
    # Source recovery picks the first-env user per prototype.
    assert kw["sources"] == [
        "/World/envs/env_0/Object",
        "/World/envs/env_1/Object",
        "/World/envs/env_0/Robot",
    ]
    assert kw["destinations"] == ["/World/envs/env_{}/Object", "/World/envs/env_{}/Object", "/World/envs/env_{}/Robot"]
    assert kw["mapping"].shape == (3, 4)
    assert kw["positions"].shape == (4, 3)


def test_build_from_clone_plans_missing_sets_error_state(stub_provider):
    """When no clone plans are published, model/state stay unset."""
    stub_provider._simulation_context = SimpleNamespace(get_clone_plans=lambda: {})
    stub_provider._stage = object()

    stub_provider._build_newton_model_from_clone_plans()

    assert stub_provider._last_newton_model_build_source == "missing"
    assert stub_provider._newton_model is None
    assert stub_provider._newton_state is None


def test_build_from_clone_plans_skips_unused_prototype_rows(stub_provider, newton_stub):
    """A prototype row with no assigned env (all-False mask row) is dropped, not raised on.

    When ``num_prototypes > num_envs`` under a sequential strategy (or any strategy that
    leaves some prototypes unused), ``clone_mask[row].nonzero()[0]`` would otherwise raise
    ``IndexError``. The provider must filter unused rows out of sources/destinations/mask.
    """
    # 3 prototypes, 2 envs, sequential: env 0 → proto 0, env 1 → proto 1, proto 2 unused.
    plans = {
        "/World/envs/env_{}/Object": ClonePlan(
            dest_template="/World/envs/env_{}/Object",
            prototype_paths=[
                "/World/template/Object/proto_0",
                "/World/template/Object/proto_1",
                "/World/template/Object/proto_2",
            ],
            clone_mask=torch.tensor([[True, False], [False, True], [False, False]], dtype=torch.bool),
        )
    }
    stub_provider._simulation_context = SimpleNamespace(get_clone_plans=lambda: plans)
    stub_provider._stage = _silent_stage()

    stub_provider._build_newton_model_from_clone_plans()

    assert stub_provider._last_newton_model_build_source == "built"
    kw = newton_stub.calls[-1]
    # Unused proto_2 row dropped; only the two assigned prototypes survive.
    assert kw["sources"] == ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    assert kw["mapping"].shape == (2, 2)


def test_build_from_clone_plans_uses_dest_template_for_env_lookup(stub_provider, newton_stub):
    """Env-origin lookup uses the per-plan ``dest_template`` prefix, not a hardcoded path.

    A scene with a non-default env path (``/Stage/scenes/env_<i>``) should still have its
    xform translates read correctly. Replaces the prior hardcoded ``/World/envs/env_<i>``.
    """
    visited: list[str] = []

    def _get_prim(path):
        visited.append(path)
        return SimpleNamespace(IsValid=lambda: False)

    plans = {
        "/Stage/scenes/env_{}/Object": ClonePlan(
            dest_template="/Stage/scenes/env_{}/Object",
            prototype_paths=["/Stage/template/Object/proto_0"],
            clone_mask=torch.ones((1, 3), dtype=torch.bool),
        )
    }
    stub_provider._simulation_context = SimpleNamespace(get_clone_plans=lambda: plans)
    stub_provider._stage = SimpleNamespace(GetPrimAtPath=_get_prim)

    stub_provider._build_newton_model_from_clone_plans()

    assert {f"/Stage/scenes/env_{i}" for i in range(3)} <= set(visited)
    assert not any(p.startswith("/World/envs/") for p in visited)


def test_clone_plan_is_hashable_with_unhashable_fields():
    """``ClonePlan`` must hash despite carrying a tensor and a list.

    With ``field(hash=False)`` on the unhashable members, hashing operates on
    ``dest_template`` only — the natural identity (it is the dict key in
    :meth:`SimulationContext.get_clone_plans`).
    """
    plan_a = ClonePlan(
        dest_template="/World/envs/env_{}/Object",
        prototype_paths=["/World/template/Object/proto_0"],
        clone_mask=torch.ones((1, 4), dtype=torch.bool),
    )
    plan_b = ClonePlan(
        dest_template="/World/envs/env_{}/Object",
        prototype_paths=["/World/template/Object/proto_99"],
        clone_mask=torch.zeros((1, 4), dtype=torch.bool),
    )
    assert isinstance(hash(plan_a), int)
    # Equality folds in only dest_template, so two plans with the same destination compare
    # equal regardless of prototype/mask differences.
    assert plan_a == plan_b
