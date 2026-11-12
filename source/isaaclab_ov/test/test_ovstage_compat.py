# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Population requirements from physics and cameras must compose across native API versions."""

import sys
from types import SimpleNamespace

import pytest
from isaaclab_ov import ovstage_compat

pytestmark = [pytest.mark.kitless]


@pytest.mark.parametrize("physics, legacy", [(False, False), (True, False), (True, True)])
def test_camera_population_preserves_rendering_and_physics_requirements(monkeypatch, physics, legacy):
    calls = []
    camera_desc, physics_desc = object(), object()
    domains = SimpleNamespace(RENDERING=1)
    if legacy:
        domains.PHYSICS = 2
    native = SimpleNamespace(
        PopulationDomain=domains,
        population=SimpleNamespace(
            DEFAULT_TIME=0.0,
            Desc=lambda **kwargs: SimpleNamespace(**kwargs),
            open_usd_from_string_with_desc=lambda stage, usd, **kwargs: calls.append(kwargs),
        ),
    )
    monkeypatch.setitem(sys.modules, "ovstage", native)
    monkeypatch.setattr(ovstage_compat, "_physics_population_desc", lambda: physics_desc)
    ovstage_compat.populate_stage(
        object(), "scene", ordinal=17, domains=1, physics=physics, population_desc=camera_desc
    )
    assert len(calls) == 1
    assert calls[0]["ordinal"] == 17
    descriptors = calls[0]["descs"]
    assert descriptors[0].domains == (3 if physics and legacy else 1)
    assert descriptors[-1] is camera_desc
    assert (physics_desc in descriptors) is (physics and not legacy)
