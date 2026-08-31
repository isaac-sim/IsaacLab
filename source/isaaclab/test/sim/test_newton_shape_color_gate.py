"""Regression test for the Newton shape-color-replacement gate.

The deprecated ``replace_newton_builder_shape_colors`` workaround traverses USD material
bindings; on assets with malformed/out-of-scope PhysicsMaterial bindings (e.g. ShadowHand)
it can corrupt the heap and SIGABRT during env cloning. This gate lets it be disabled via
``ISAACLAB_NEWTON_REPLACE_SHAPE_COLORS=0``.
"""
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors


class _FakeBuilder:
    def __init__(self):
        self.shape_label = ["/World/env_0/robot/geom_a", "/World/env_0/robot/geom_b"]
        self.shape_color = [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2)]


def test_gate_off_skips_replacement(monkeypatch):
    monkeypatch.setenv("ISAACLAB_NEWTON_REPLACE_SHAPE_COLORS", "0")
    b = _FakeBuilder()
    before = list(b.shape_color)
    n = replace_newton_builder_shape_colors(b, stage=None)  # stage untouched when gated off
    assert n == 0
    assert b.shape_color == before


def test_gate_on_by_default(monkeypatch):
    monkeypatch.delenv("ISAACLAB_NEWTON_REPLACE_SHAPE_COLORS", raising=False)
    import isaaclab.sim.utils.newton_model_utils as m
    monkeypatch.setattr(m, "_resolve_shape_color", lambda *a, **k: None)
    b = _FakeBuilder()
    before = list(b.shape_color)
    n = replace_newton_builder_shape_colors(b, stage=object())
    assert n == 0
    assert b.shape_color == before
