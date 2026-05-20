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


def test_load_importer_api_prefers_isaac_sim_extension(monkeypatch):
    """The Isaac Sim extension path should win when both providers are available."""
    calls = []

    monkeypatch.setattr(importer_api, "_is_isaacsim_available", lambda: True)
    monkeypatch.setattr(importer_api, "_get_standalone_importer_distribution_path", lambda: "/standalone/site-packages")
    monkeypatch.setattr(
        importer_api,
        "_enable_isaacsim_extension",
        lambda extension_name: calls.append(("enable", extension_name)),
    )
    monkeypatch.setattr(
        importer_api.importlib,
        "import_module",
        lambda module_name: calls.append(("import", module_name)) or _make_importer_module("urdf"),
    )
    monkeypatch.setattr(
        importer_api,
        "_import_standalone_importer_module",
        lambda module_name, distribution_path: pytest.fail("Unexpected standalone importer load"),
    )

    importer_cls, config_cls = importer_api.load_importer_api("urdf")

    assert importer_cls is _Importer
    assert config_cls is _ImporterConfig
    assert calls == [
        ("enable", "isaacsim.asset.importer.urdf"),
        ("import", "isaacsim.asset.importer.urdf"),
    ]


def test_load_importer_api_uses_isaac_sim_extension_without_standalone(monkeypatch):
    """Missing standalone package should use the extension-backed import path."""
    calls = []

    monkeypatch.setattr(importer_api, "_is_isaacsim_available", lambda: True)
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


def test_load_importer_api_falls_back_to_standalone_package(monkeypatch):
    """A missing Isaac Sim runtime should fall back to the standalone package."""
    calls = []

    monkeypatch.setattr(importer_api, "_is_isaacsim_available", lambda: False)
    monkeypatch.setattr(importer_api, "_get_standalone_importer_distribution_path", lambda: "/standalone/site-packages")
    monkeypatch.setattr(
        importer_api,
        "_enable_isaacsim_extension",
        lambda extension_name: pytest.fail("Unexpected Isaac Sim extension load"),
    )
    monkeypatch.setattr(
        importer_api,
        "_import_standalone_importer_module",
        lambda module_name, distribution_path: calls.append(("standalone", module_name, distribution_path))
        or _make_importer_module("urdf"),
    )

    importer_cls, config_cls = importer_api.load_importer_api("urdf")

    assert importer_cls is _Importer
    assert config_cls is _ImporterConfig
    assert calls == [
        ("standalone", "isaacsim.asset.importer.urdf", "/standalone/site-packages"),
    ]


def test_load_importer_api_does_not_fallback_when_isaac_sim_extension_fails(monkeypatch):
    """An available Isaac Sim runtime should not fall back to the standalone package."""
    calls = []

    monkeypatch.setattr(importer_api, "_is_isaacsim_available", lambda: True)
    monkeypatch.setattr(importer_api, "_get_standalone_importer_distribution_path", lambda: "/standalone/site-packages")
    monkeypatch.setattr(
        importer_api,
        "_enable_isaacsim_extension",
        lambda extension_name: calls.append(("enable", extension_name)),
    )
    monkeypatch.setattr(
        importer_api.importlib,
        "import_module",
        lambda module_name: (_ for _ in ()).throw(ImportError(f"failed to import {module_name}")),
    )
    monkeypatch.setattr(
        importer_api,
        "_import_standalone_importer_module",
        lambda module_name, distribution_path: pytest.fail("Unexpected standalone importer load"),
    )

    with pytest.raises(ImportError, match="standalone .* package will not be used"):
        importer_api.load_importer_api("mjcf")

    assert calls == [("enable", "isaacsim.asset.importer.mjcf")]
