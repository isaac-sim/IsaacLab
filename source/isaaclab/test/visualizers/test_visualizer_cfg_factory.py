"""Unit tests for visualizer config factory behavior."""

from __future__ import annotations

import pytest

import isaaclab.visualizers as visualizers_module
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


def test_create_visualizer_raises_for_base_cfg():
    cfg = VisualizerCfg()
    with pytest.raises(ValueError, match="Cannot create visualizer from base VisualizerCfg class"):
        cfg.create_visualizer()


def test_create_visualizer_raises_for_unknown_type(monkeypatch):
    monkeypatch.setattr(visualizers_module, "get_visualizer_class", lambda name: None)
    cfg = VisualizerCfg(visualizer_type="unknown-backend")
    with pytest.raises(ValueError, match="not registered"):
        cfg.create_visualizer()


def test_create_visualizer_raises_import_error_for_newton_family(monkeypatch):
    monkeypatch.setattr(visualizers_module, "get_visualizer_class", lambda name: None)
    cfg = VisualizerCfg(visualizer_type="newton")
    with pytest.raises(ImportError, match="requires the Newton Python module"):
        cfg.create_visualizer()
