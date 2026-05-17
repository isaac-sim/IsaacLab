# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import types

import pytest

from isaaclab.sim.converters import _importer_api as importer_api


class _Importer:
    pass


class _ImporterConfig:
    pass


def _make_importer_module(importer_kind: importer_api.ImporterKind) -> types.SimpleNamespace:
    if importer_kind == "mjcf":
        return types.SimpleNamespace(MJCFImporter=_Importer, MJCFImporterConfig=_ImporterConfig)
    return types.SimpleNamespace(URDFImporter=_Importer, URDFImporterConfig=_ImporterConfig)


def test_load_importer_api_prefers_standalone_package(monkeypatch):
    """The standalone package path should not touch the Isaac Sim extension manager."""
    calls = []

    monkeypatch.setattr(importer_api, "_get_standalone_importer_distribution_path", lambda: "/standalone/site-packages")
    monkeypatch.setattr(
        importer_api.importlib,
        "import_module",
        lambda module_name: calls.append(("import", module_name)) or _make_importer_module("urdf"),
    )
    monkeypatch.setattr(
        importer_api,
        "_enable_isaacsim_extension",
        lambda extension_name: pytest.fail(f"Unexpected extension enable: {extension_name}"),
    )

    importer_cls, config_cls = importer_api.load_importer_api("urdf")

    assert importer_cls is _Importer
    assert config_cls is _ImporterConfig
    assert calls == [("import", "isaacsim.asset.importer.urdf")]


def test_load_importer_api_falls_back_to_isaac_sim_extension(monkeypatch):
    """Missing standalone package should preserve the extension-backed import path."""
    calls = []

    monkeypatch.setattr(importer_api, "_get_standalone_importer_distribution_path", lambda: None)
    monkeypatch.setattr(
        importer_api,
        "_enable_isaacsim_extension",
        lambda extension_name: calls.append(("enable", extension_name)),
    )
    monkeypatch.setattr(
        importer_api.importlib,
        "import_module",
        lambda module_name: calls.append(("import", module_name)) or _make_importer_module("mjcf"),
    )

    importer_cls, config_cls = importer_api.load_importer_api("mjcf")

    assert importer_cls is _Importer
    assert config_cls is _ImporterConfig
    assert calls == [
        ("enable", "isaacsim.asset.importer.mjcf"),
        ("import", "isaacsim.asset.importer.mjcf"),
    ]


def test_load_importer_api_falls_back_when_standalone_import_fails(monkeypatch):
    """A broken standalone import should still allow the existing extension path to work."""
    calls = []

    def _import_module(module_name):
        calls.append(("import", module_name))
        if len(calls) == 1:
            raise ImportError("standalone package import failed")
        return _make_importer_module("urdf")

    monkeypatch.setattr(importer_api, "_get_standalone_importer_distribution_path", lambda: "/standalone/site-packages")
    monkeypatch.setattr(importer_api.importlib, "import_module", _import_module)
    monkeypatch.setattr(
        importer_api,
        "_enable_isaacsim_extension",
        lambda extension_name: calls.append(("enable", extension_name)),
    )

    importer_cls, config_cls = importer_api.load_importer_api("urdf")

    assert importer_cls is _Importer
    assert config_cls is _ImporterConfig
    assert calls == [
        ("import", "isaacsim.asset.importer.urdf"),
        ("enable", "isaacsim.asset.importer.urdf"),
        ("import", "isaacsim.asset.importer.urdf"),
    ]
