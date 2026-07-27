# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import types

import pytest

import isaaclab.sim.utils as sim_utils
from isaaclab.sim.converters import _importer_api as importer_api


class _Importer:
    pass


class _ImporterConfig:
    pass


def _make_importer_module(importer_kind: importer_api.ImporterKind) -> types.SimpleNamespace:
    if importer_kind == "mjcf":
        return types.SimpleNamespace(MJCFImporter=_Importer, MJCFImporterConfig=_ImporterConfig)
    return types.SimpleNamespace(URDFImporter=_Importer, URDFImporterConfig=_ImporterConfig)


def test_load_api_enables_extension_when_kit_is_running(monkeypatch: pytest.MonkeyPatch):
    # under a running Kit app, load_api enables the owning extension before importing it.
    calls = []
    monkeypatch.setattr(importer_api, "has_kit", lambda: True)
    monkeypatch.setattr(sim_utils, "enable_extension", lambda extension_name: calls.append(("enable", extension_name)))
    monkeypatch.setattr(
        importer_api.importlib,
        "import_module",
        lambda module_name: calls.append(("import", module_name)) or _make_importer_module("urdf"),
    )

    importer_cls, config_cls = importer_api.ImporterProvider.load_api("urdf")

    assert importer_cls is _Importer
    assert config_cls is _ImporterConfig
    assert calls == [
        ("enable", "isaacsim.asset.importer.urdf"),
        ("import", "isaacsim.asset.importer.urdf"),
    ]


def test_load_api_imports_standalone_provider_without_kit(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(importer_api, "has_kit", lambda: False)
    monkeypatch.setattr(
        importer_api.importlib,
        "import_module",
        lambda module_name: calls.append(module_name) or _make_importer_module("mjcf"),
    )

    importer_cls, config_cls = importer_api.ImporterProvider.load_api("mjcf")

    assert importer_cls is _Importer
    assert config_cls is _ImporterConfig
    assert calls == ["isaacsim.asset.importer.mjcf"]


def test_load_api_preserves_importer_class_identity(monkeypatch: pytest.MonkeyPatch):
    # seed the real sys.modules and use the real importlib: evicting the module between
    # loads (the original regression) would bypass this entry and break class identity
    module_name = "isaacsim.asset.importer.urdf"
    sentinel_module = types.ModuleType(module_name)
    sentinel_module.URDFImporter = _Importer
    sentinel_module.URDFImporterConfig = _ImporterConfig
    monkeypatch.setattr(importer_api, "has_kit", lambda: False)
    monkeypatch.setitem(sys.modules, module_name, sentinel_module)

    first_importer, first_config = importer_api.ImporterProvider.load_api("urdf")
    second_importer, second_config = importer_api.ImporterProvider.load_api("urdf")

    assert sys.modules[module_name] is sentinel_module
    assert first_importer is second_importer is _Importer
    assert first_config is second_config is _ImporterConfig


def test_load_api_preserves_module_search_path(monkeypatch: pytest.MonkeyPatch):
    original_path = list(sys.path)
    monkeypatch.setattr(importer_api, "has_kit", lambda: False)
    monkeypatch.setattr(importer_api.importlib, "import_module", lambda module_name: _make_importer_module("urdf"))

    importer_api.ImporterProvider.load_api("urdf")

    assert sys.path == original_path


def test_load_api_reports_both_provider_options(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(importer_api, "has_kit", lambda: False)
    monkeypatch.setattr(
        importer_api.importlib,
        "import_module",
        lambda module_name: (_ for _ in ()).throw(ImportError(module_name)),
    )

    with pytest.raises(ImportError, match="Launch Isaac Sim.*standalone 'isaacsim-asset-isolated'"):
        importer_api.ImporterProvider.load_api("urdf")


def test_load_api_wraps_kit_extension_errors(monkeypatch: pytest.MonkeyPatch):
    # under a running Kit app the extension enable is attempted; a failure there is wrapped.
    monkeypatch.setattr(importer_api, "has_kit", lambda: True)
    monkeypatch.setattr(
        sim_utils,
        "enable_extension",
        lambda extension_name: (_ for _ in ()).throw(RuntimeError(extension_name)),
    )

    with pytest.raises(ImportError, match="Launch Isaac Sim.*standalone 'isaacsim-asset-isolated'"):
        importer_api.ImporterProvider.load_api("mjcf")


def test_is_standalone_available_checks_distribution_metadata(monkeypatch: pytest.MonkeyPatch):
    requested = []
    monkeypatch.setattr(
        importer_api.metadata,
        "distribution",
        lambda distribution_name: requested.append(distribution_name) or object(),
    )

    assert importer_api.ImporterProvider.is_standalone_available()
    assert requested == ["isaacsim-asset-isolated"]


def test_is_standalone_available_handles_missing_distribution(monkeypatch: pytest.MonkeyPatch):
    def _raise_missing(distribution_name: str):
        raise importer_api.metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(importer_api.metadata, "distribution", _raise_missing)

    assert not importer_api.ImporterProvider.is_standalone_available()


def test_validate_standalone_runtime_builds_import_statement(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(importer_api.subprocess, "run", _run)

    importer_api.ImporterProvider.validate_standalone_runtime("mjcf")

    assert captured["cmd"][-1] == "from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig"


def test_validate_standalone_runtime_reports_subprocess_failure(monkeypatch: pytest.MonkeyPatch):
    result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="missing dependency")
    monkeypatch.setattr(importer_api.subprocess, "run", lambda *args, **kwargs: result)

    with pytest.raises(ImportError, match="missing dependency"):
        importer_api.ImporterProvider.validate_standalone_runtime("mjcf")


def test_validate_standalone_runtime_reports_timeout(monkeypatch: pytest.MonkeyPatch):
    def _time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(importer_api.subprocess, "run", _time_out)

    with pytest.raises(ImportError, match="did not load within"):
        importer_api.ImporterProvider.validate_standalone_runtime("mjcf", timeout=5.0)
