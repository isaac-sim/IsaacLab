# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""OVPhysX actuator-control unit tests."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.assets.articulation import actuator_control  # noqa: E402
from isaaclab_ov.assets.articulation.actuator_control import OvPhysxActuatorControl  # noqa: E402

from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402

pytestmark = pytest.mark.unit


def test_prepare_native_actuators_leaves_implicit_only_articulation_on_standard_path(monkeypatch):
    """Keep implicit-only articulations on the unchanged solver-drive path."""
    runtime_prepare_calls = []
    runtime = SimpleNamespace(
        prepare=lambda *args, **kwargs: runtime_prepare_calls.append(True), wrapper=None, adapter=None
    )
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        cfg=SimpleNamespace(prim_path="/World/Robot"),
    )
    monkeypatch.setattr(actuator_control, "PhysxActuatorRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(actuator_control, "find_first_matching_prim", lambda _: None)

    control = OvPhysxActuatorControl(articulation)
    native_groups = control.prepare_native_actuators(
        collection=None,
        actuator_cfgs={"implicit": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0)},
    )

    assert native_groups == set()
    assert not control.native_actuator_path_active
    assert not articulation._has_newton_actuators
    assert runtime_prepare_calls == []


@pytest.mark.parametrize(
    "module_name",
    [
        "isaaclab_physx.assets.articulation.actuator_control",
        "isaaclab_ov.assets.articulation.actuator_control",
    ],
)
def test_host_actuator_control_import_does_not_probe_optional_newton_runtime(monkeypatch, module_name):
    """Import host controls without probing an unrequested Newton optional dependency."""
    original_find_spec = importlib.util.find_spec

    def reject_newton_probe(name, *args, **kwargs):
        if name.startswith("isaaclab_newton"):
            raise AssertionError("host actuator-control import eagerly probed Newton")
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", reject_newton_probe)
    importlib.reload(importlib.import_module(module_name))
