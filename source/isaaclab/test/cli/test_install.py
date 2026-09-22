# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``isaaclab -i`` install command.

Package managers are never invoked: pip commands are recorded through ``run_command`` and the
filesystem effects (prebundle repointing, symlink integrity) run against temporary trees.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from isaaclab.cli.commands import install

pytestmark = pytest.mark.unit

_PIP = ["/env/bin/python", "-m", "pip"]
_UV_PIP = ["uv", "pip"]


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


@pytest.fixture
def run_command(monkeypatch):
    """Record the commands ``install`` would run and answer each with a successful, empty result."""
    commands: list[list[str]] = []

    def _run(cmd, **_kwargs):
        commands.append(list(cmd))
        return _completed()

    monkeypatch.setattr(install, "run_command", _run)
    monkeypatch.setattr(install, "extract_python_exe", lambda: _PIP[0])
    return commands


# ---------------------------------------------------------------------------
# Token parsing and constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("install_type", "expected"),
    [
        ("newton", ["newton"]),
        ("newton , mimic", ["newton", "mimic"]),
        ("rl[rsl-rl,skrl],mimic", ["rl[rsl-rl,skrl]", "mimic"]),
        ("a[b[c,d],e],f", ["a[b[c,d],e]", "f"]),
        ("assets,tasks,", ["assets", "tasks"]),
        ("", []),
        ("rl[rsl-rl", ["rl[rsl-rl"]),
    ],
)
def test_split_install_items(install_type, expected):
    """Commas split tokens except inside brackets; whitespace and empty tokens are dropped."""
    assert install.split_install_items(install_type) == expected


def test_install_constants_are_consistent():
    """The submodule and feature tables must not overlap and ``isaaclab`` must install first."""
    optional_packages = {pkg for pkgs in install.OPTIONAL_ISAACLAB_SUBMODULES.values() for pkg in pkgs}
    assert install.CORE_ISAACLAB_SUBMODULES[0] == "isaaclab"
    assert optional_packages.isdisjoint(install.CORE_ISAACLAB_SUBMODULES)
    assert set(install.OPTIONAL_ISAACLAB_SUBMODULES).isdisjoint(install.VALID_EXTRA_FEATURES)
    assert install.MANUAL_EXTRA_FEATURES <= install.VALID_EXTRA_FEATURES
    # each optional source directory maps to the same-named root extra
    assert {
        name: tuple(pkg.removeprefix("isaaclab_") for pkg in pkgs)
        for name, pkgs in install.OPTIONAL_ISAACLAB_SUBMODULES.items()
    } == install.OPTIONAL_SUBMODULE_ROOT_EXTRAS


# ---------------------------------------------------------------------------
# command_install dispatch
# ---------------------------------------------------------------------------

_STEPS = (
    "_install_system_deps",
    "_install_isaaclab_submodules",
    "_install_extra_feature",
    "_install_optional_submodule_extra_dependencies",
    "_install_centralized_dependencies",
    "_install_isaacsim",
    "_ensure_cuda_torch",
    "_ensure_newton",
    "_maybe_uninstall_prebundled_torch",
    "_ensure_pink_ik_dependencies_installed",
    "_repoint_prebundle_packages",
    "_find_dangling_prebundle_symlinks",
    "_assert_no_new_dangling_prebundle_symlinks",
    "command_editor",
)

_OPTIONAL_PACKAGES = [pkg for pkgs in install.OPTIONAL_ISAACLAB_SUBMODULES.values() for pkg in pkgs]
_AUTO_FEATURES = {(name, "") for name in install.VALID_EXTRA_FEATURES - install.MANUAL_EXTRA_FEATURES}


@pytest.fixture
def install_steps(monkeypatch, run_command):
    """Replace every installation step of ``command_install`` with a recording mock."""
    steps = {name: mock.Mock(name=name) for name in _STEPS}
    for name, step in steps.items():
        monkeypatch.setattr(install, name, step)
    steps["_find_dangling_prebundle_symlinks"].return_value = set()
    monkeypatch.setattr(install, "get_pip_command", lambda _python: list(_PIP))
    return steps


@pytest.mark.parametrize(
    ("install_type", "optional_names", "features", "install_isaacsim", "optional_selectors"),
    [
        ("all", list(install.OPTIONAL_ISAACLAB_SUBMODULES), _AUTO_FEATURES, False, []),
        ("core", [], set(), False, []),
        ("none", [], set(), False, []),
        ("newton", [], {("newton", "")}, False, []),
        ("rl[rsl-rl]", [], {("rl", "rsl-rl")}, False, []),
        ("contrib[rlinf],ov", [], {("contrib", "rlinf"), ("ov", "")}, False, []),
        ("mimic,teleop", ["mimic", "teleop"], set(), False, []),
        ("teleop[foo],newton[sim]", ["teleop"], {("newton", "sim")}, False, [("teleop", "foo")]),
        ("isaacsim", [], set(), True, []),
    ],
)
def test_command_install_dispatch(
    install_steps, install_type, optional_names, features, install_isaacsim, optional_selectors
):
    """The install value selects optional submodules, extra features, and the Isaac Sim install."""
    install.command_install(install_type)

    optional_packages = list(
        dict.fromkeys(pkg for name in optional_names for pkg in install.OPTIONAL_ISAACLAB_SUBMODULES[name])
    )
    installed = install_steps["_install_isaaclab_submodules"].call_args.args[0]
    assert installed == [*install.CORE_ISAACLAB_SUBMODULES, *optional_packages]
    assert install_steps["_install_centralized_dependencies"].call_args.args[1] == optional_names
    assert {call.args for call in install_steps["_install_extra_feature"].call_args_list} == features
    assert install_steps["_install_isaacsim"].called is install_isaacsim
    selectors = install_steps["_install_optional_submodule_extra_dependencies"].call_args_list
    assert [call.args for call in selectors] == optional_selectors
    # the integrity check compares against the baseline taken before any pip pass
    assert install_steps["_assert_no_new_dangling_prebundle_symlinks"].call_args.args == (set(),)


@pytest.mark.parametrize("token", ["rl[rsl-rl", "totally_unknown_package"])
def test_command_install_warns_and_skips_bad_tokens(install_steps, monkeypatch, token):
    """Malformed and unknown tokens are reported once and leave the core install untouched."""
    warnings = []
    monkeypatch.setattr(install, "print_warning", warnings.append)

    install.command_install(token)

    assert len(warnings) == 1 and token in warnings[0]
    assert install_steps["_install_isaaclab_submodules"].call_args.args[0] == install.CORE_ISAACLAB_SUBMODULES
    install_steps["_install_extra_feature"].assert_not_called()


def test_command_install_filters_prebundles_from_pythonpath_and_probes_with_the_original(
    install_steps, monkeypatch, run_command
):
    """pip runs without Isaac Sim's prebundle paths, while runtime probes still see them."""
    original = os.pathsep.join(["/env/site-packages", "/isaac/exts/ml_archive/pip_prebundle"])
    monkeypatch.setenv("PYTHONPATH", original)
    monkeypatch.setenv("LD_PRELOAD", "libgomp.so.1")
    monkeypatch.setattr(install, "is_arm", lambda: True)
    seen: dict[str, str | None] = {}

    def _probe(_python, _pip, *, probe_env):
        seen["pythonpath"] = os.environ.get("PYTHONPATH")
        seen["ld_preload"] = os.environ.get("LD_PRELOAD")
        seen["cmake_policy"] = os.environ.get("CMAKE_POLICY_VERSION_MINIMUM")
        seen["probe_pythonpath"] = probe_env["PYTHONPATH"]

    install_steps["_maybe_uninstall_prebundled_torch"].side_effect = _probe

    install.command_install("core")

    assert seen == {
        "pythonpath": "/env/site-packages",
        "ld_preload": None,
        "cmake_policy": "3.5",
        "probe_pythonpath": original,
    }
    assert os.environ["PYTHONPATH"] == original
    assert os.environ["LD_PRELOAD"] == "libgomp.so.1"
    assert "CMAKE_POLICY_VERSION_MINIMUM" not in os.environ
    # pip itself is upgraded and setuptools pinned before anything else is installed
    assert run_command[:2] == [[*_PIP, "install", "--upgrade", "pip"], [*_PIP, "install", "setuptools<82.0.0"]]


@pytest.mark.parametrize(
    ("is_arm", "existing", "expected_after"),
    [(True, None, None), (True, "3.10", "3.10"), (False, None, None)],
)
def test_arm_cmake_policy_compatibility(monkeypatch, is_arm, existing, expected_after):
    """ARM installs pin CMake's minimum policy version for the duration of the install only."""
    monkeypatch.setattr(install, "is_arm", lambda: is_arm)
    monkeypatch.delenv("CMAKE_POLICY_VERSION_MINIMUM", raising=False)
    if existing is not None:
        monkeypatch.setenv("CMAKE_POLICY_VERSION_MINIMUM", existing)

    with install._arm_cmake_policy_compatibility():
        assert os.environ.get("CMAKE_POLICY_VERSION_MINIMUM") == ("3.5" if is_arm else None)
    assert os.environ.get("CMAKE_POLICY_VERSION_MINIMUM") == expected_after


@pytest.mark.parametrize(
    ("feature", "selector", "expected_extras"),
    [
        ("tetrahedralization", "", ["tetrahedralization"]),
        ("rl", "rsl-rl", ["rsl-rl"]),
        ("rl", "", ["rl-games", "rsl-rl", "sb3", "skrl"]),
        ("visualizer", "rerun", ["rerun"]),
        ("visualizer", "", ["rerun", "viser"]),
        ("ov", "ovrtx", ["ovrtx"]),
        ("ov", "all", ["ovrtx", "ovphysx"]),
        ("contrib", "rlinf", ["rlinf"]),
        ("contrib", "", []),
        ("newton", "", []),
    ],
)
def test_install_extra_feature_maps_to_root_extras(monkeypatch, feature, selector, expected_extras):
    """Each feature token installs the matching root ``optional-dependencies`` groups."""
    extras = []
    monkeypatch.setattr(install, "_install_root_extra", extras.append)

    install._install_extra_feature(feature, selector)

    assert extras == expected_extras


# ---------------------------------------------------------------------------
# Submodule installs and targeted dependency upgrades
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pip_cmd", "upgrade_args"),
    [(_PIP, ["--upgrade"]), (_UV_PIP, ["--upgrade-package", "isaacteleop"])],
    ids=["pip", "uv"],
)
def test_install_submodules_upgrades_declared_dependencies(monkeypatch, run_command, tmp_path, pip_cmd, upgrade_args):
    """An editable install is followed by targeted upgrades of the dependencies a submodule opts into."""
    extension_dir = tmp_path / "source" / "isaaclab_teleop"
    extension_dir.mkdir(parents=True)
    (extension_dir / "setup.py").touch()
    (extension_dir / "pyproject.toml").write_text('[tool.isaaclab]\npip_upgrade_dependencies = ["isaacteleop"]\n')
    requirement = 'isaacteleop[cloudxr] ~=1.2.0; platform_system == "Linux"'
    monkeypatch.setattr(install, "ISAACLAB_ROOT", tmp_path)
    monkeypatch.setattr(install, "get_pip_command", lambda _python: list(pip_cmd))
    monkeypatch.setattr(install, "_get_installed_distribution_requirements", lambda _python, _dist: [requirement])

    install._install_isaaclab_submodules(["isaaclab_teleop", "missing_submodule"])

    assert run_command == [
        [*pip_cmd, "install", "--editable", str(extension_dir)],
        [*pip_cmd, "install", *upgrade_args, requirement],
    ]


@pytest.mark.parametrize(
    ("dependency_names", "requirements", "expected_upgrades", "expected_warnings"),
    [
        # every matching metadata requirement is upgraded, name duplicates collapse to one
        (
            ["example-package", "Example_Package"],
            ['example-package>=1.0; platform_system == "Linux"', 'example_package>=2.0; platform_system == "Windows"'],
            ['example-package>=1.0; platform_system == "Linux"', 'example_package>=2.0; platform_system == "Windows"'],
            0,
        ),
        (["isaacteleop"], ["dex-retargeting==0.5.0"], [], 1),
        ([], [], [], 0),
    ],
)
def test_upgrade_extension_pip_dependencies(
    monkeypatch, run_command, dependency_names, requirements, expected_upgrades, expected_warnings
):
    """Targeted upgrades follow installed metadata and warn about names that are not declared there."""
    warnings = []
    monkeypatch.setattr(install, "print_warning", warnings.append)
    monkeypatch.setattr(install, "_get_installed_distribution_requirements", lambda _python, _dist: requirements)

    install._upgrade_extension_pip_dependencies(_PIP[0], _PIP, "isaaclab_teleop", dependency_names)

    assert run_command == [[*_PIP, "install", "--upgrade", requirement] for requirement in expected_upgrades]
    assert len(warnings) == expected_warnings


@pytest.mark.parametrize(
    ("pyproject", "expected", "warns"),
    [
        ('[tool.isaaclab]\npip_upgrade_dependencies = ["isaacteleop"]\n', ["isaacteleop"], False),
        ("[tool.isaaclab]\n", [], False),
        ('[tool.isaaclab]\npip_upgrade_dependencies = "isaacteleop"\n', [], True),
    ],
)
def test_extension_pip_upgrade_dependencies_are_read_from_pyproject(monkeypatch, tmp_path, pyproject, expected, warns):
    """The opt-in list must be a list of strings; anything else disables targeted upgrades with a warning."""
    (tmp_path / "pyproject.toml").write_text(pyproject)
    warnings = []
    monkeypatch.setattr(install, "print_warning", warnings.append)

    assert install._get_extension_pip_upgrade_dependencies(tmp_path) == expected
    assert bool(warnings) is warns


# ---------------------------------------------------------------------------
# torch handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prebundle_first", [True, False, None], ids=["prebundle-first", "site-first", "no-torch"])
def test_torch_first_on_sys_path_is_prebundle(tmp_path, prebundle_first):
    """The probe reports whether the first importable ``torch`` lives under a ``pip_prebundle`` directory."""
    prebundle = tmp_path / "exts" / "omni.isaac.ml_archive" / "pip_prebundle"
    site = tmp_path / "site-packages"
    for root in (prebundle, site):
        (root / "torch").mkdir(parents=True)
        (root / "torch" / "__init__.py").touch()
    if prebundle_first is None:
        search_path = [str(tmp_path / "empty")]
    else:
        search_path = [str(prebundle), str(site)] if prebundle_first else [str(site), str(prebundle)]

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(search_path)}
    assert install._torch_first_on_sys_path_is_prebundle(sys.executable, env=env) is bool(prebundle_first)


@pytest.mark.parametrize(
    ("shadowed", "pip_cmd", "expected"),
    [
        (False, _PIP, []),
        (True, _PIP, [[*_PIP, "uninstall", "-y", "torch", "torchvision", "torchaudio"]]),
        (True, _UV_PIP, [[*_UV_PIP, "uninstall", "torch", "torchvision", "torchaudio"]]),
    ],
    ids=["not-shadowed", "pip", "uv"],
)
def test_maybe_uninstall_prebundled_torch(monkeypatch, run_command, shadowed, pip_cmd, expected):
    """A shadowing prebundle triggers a torch-stack uninstall; ``uv pip`` takes no ``-y``."""
    probes = []
    monkeypatch.setattr(
        install, "_torch_first_on_sys_path_is_prebundle", lambda python, *, env: probes.append(env) or shadowed
    )

    install._maybe_uninstall_prebundled_torch(pip_cmd[0], pip_cmd, probe_env={"PYTHONPATH": "/orig"})

    assert probes == [{"PYTHONPATH": "/orig"}]
    assert run_command == expected


@pytest.mark.parametrize(
    ("pip_cmd", "installed_version", "expect_install"),
    [
        (_PIP, "2.12.0+cu130", False),
        (_PIP, "2.12.0+cu126", True),
        (_UV_PIP, None, True),
    ],
    ids=["matching", "wrong-cuda-tag", "missing-uv"],
)
def test_ensure_cuda_torch(monkeypatch, run_command, pip_cmd, installed_version, expect_install):
    """torch is reinstalled from the CUDA 13 index unless the pinned build tag is already present."""
    monkeypatch.setattr(install, "get_pip_command", lambda _python: list(pip_cmd))
    monkeypatch.setattr(install, "_pinned_version", {"torch": "2.12.0", "torchvision": "0.27.0"}.__getitem__)
    show = _completed(0, f"Name: torch\nVersion: {installed_version}\n") if installed_version else _completed(1)
    monkeypatch.setattr(install, "run_command", lambda cmd, **_: run_command.append(list(cmd)) or show)

    install._ensure_cuda_torch()

    assert run_command[0] == [*pip_cmd, "show", "torch"]
    if not expect_install:
        assert len(run_command) == 1
        return
    confirm = [] if pip_cmd == _UV_PIP else ["-y"]
    assert run_command[1:] == [
        [*pip_cmd, "uninstall", *confirm, "torch", "torchvision", "torchaudio"],
        [
            *pip_cmd,
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
            "torch==2.12.0",
            "torchvision==0.27.0",
        ],
    ]


# ---------------------------------------------------------------------------
# Newton, Isaac Sim, and root-extra installs
# ---------------------------------------------------------------------------


def test_ensure_newton_installs_pinned_release_when_absent(monkeypatch, run_command):
    """The Newton pin from ``[tool.uv].override-dependencies`` replaces whatever Isaac Sim bundled."""
    overrides = install._load_root_pyproject()["tool"]["uv"]["override-dependencies"]
    requirement = next(r for r in overrides if install._requirement_name(r) == "newton")
    schemas = next(r for r in overrides if install._requirement_name(r) == "newton-usd-schemas")
    monkeypatch.setattr(install, "get_pip_command", lambda _python: list(_UV_PIP))
    monkeypatch.setattr(
        install,
        "run_command",
        lambda cmd, **_: run_command.append(list(cmd)) or _completed(0, "numpy==2.0.0\n"),
    )

    install._ensure_newton()

    assert run_command == [
        [*_UV_PIP, "freeze"],
        [*_UV_PIP, "uninstall", "newton"],
        [*_UV_PIP, "install", requirement, schemas],
    ]


@pytest.mark.parametrize(
    ("requirement", "freeze_line"),
    [
        ("newton[sim]==1.5.1", "newton==1.5.1"),
        (
            "newton[sim] @ git+https://github.com/newton-physics/newton.git@cca3bb8",
            "newton @ git+https://github.com/newton-physics/newton.git@cca3bb8",
        ),
    ],
)
def test_ensure_newton_skips_installed_pin(monkeypatch, run_command, requirement, freeze_line):
    """Release and Git-revision pins reported by ``pip freeze`` are left alone."""
    monkeypatch.setattr(
        install, "_load_root_pyproject", lambda: {"tool": {"uv": {"override-dependencies": [requirement]}}}
    )
    monkeypatch.setattr(install, "get_pip_command", lambda _python: list(_UV_PIP))
    monkeypatch.setattr(
        install, "run_command", lambda cmd, **_: run_command.append(list(cmd)) or _completed(0, f"{freeze_line}\n")
    )

    install._ensure_newton()

    assert run_command == [[*_UV_PIP, "freeze"]]


@pytest.mark.parametrize(
    ("results", "expected_requirement"),
    [
        # not installed: use the root pyproject pin
        ([_completed(1)], install._isaacsim_requirement()),
        # kernel-only install: complete it at the installed version
        ([_completed(0, "1.2.3+local"), _completed(1)], "isaacsim[all,extscache]==1.2.3+local"),
        # full runtime present: nothing to install
        ([_completed(0, "1.2.3"), _completed(0)], None),
    ],
    ids=["missing", "kernel-only", "installed"],
)
def test_install_isaacsim(monkeypatch, run_command, results, expected_requirement):
    """Isaac Sim is installed from the NVIDIA index with uv's cross-index resolution strategy."""
    answers = iter(results)
    monkeypatch.setattr(install, "get_pip_command", lambda _python: list(_UV_PIP))
    monkeypatch.setattr(
        install, "run_command", lambda cmd, **_: run_command.append(list(cmd)) or next(answers, _completed())
    )

    install._install_isaacsim()

    installs = [cmd for cmd in run_command if cmd[:3] == [*_UV_PIP, "install"]]
    if expected_requirement is None:
        assert installs == []
    else:
        assert installs == [
            [
                *_UV_PIP,
                "install",
                expected_requirement,
                "--extra-index-url",
                install.NVIDIA_INDEX_URL,
                "--index-strategy",
                "unsafe-best-match",
            ]
        ]


def test_install_root_extra_excludes_isaacsim(monkeypatch, run_command):
    """pip cannot resolve Isaac Sim beside Isaac Teleop in one pass, so the ``teleop`` extra omits it."""
    monkeypatch.setattr(install, "get_pip_command", lambda _python: list(_PIP))

    install._install_root_extra("teleop")

    (command,) = run_command
    assert command[:4] == [*_PIP, "install"]
    assert not any(dep.startswith("isaacsim") for dep in command[4:])
    assert any(dep.startswith("isaacteleop") for dep in command[4:])


def test_pink_ik_stack_is_derived_from_root_pins(source_checkout_root: Path):
    """The Pink IK force-install stack mirrors the exact pins of the root ``pyproject.toml``."""
    with mock.patch.object(install, "ISAACLAB_ROOT", source_checkout_root):
        stack = install._pink_ik_stack()

    assert [install._requirement_name(r) for r in stack] == list(install._PINK_IK_PACKAGES)
    assert all(";" not in r for r in stack), "environment markers must be stripped"
    assert all(any(r.startswith(f"{name}==") for r in stack) for name in ("pin-pink", "daqp"))


# ---------------------------------------------------------------------------
# Prebundle repointing
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_isaac_sim(monkeypatch, tmp_path):
    """A fake Isaac Sim tree with one ``pip_prebundle`` directory and a fake ``site-packages``."""
    isaacsim_path = tmp_path / "isaac_sim"
    prebundle = isaacsim_path / "exts" / "some.ext" / "pip_prebundle"
    prebundle.mkdir(parents=True)
    site_packages = tmp_path / "env" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr(install, "extract_isaacsim_path", lambda **_: isaacsim_path)
    monkeypatch.setattr(install, "extract_python_exe", lambda: str(tmp_path / "env" / "bin" / "python"))
    monkeypatch.setattr(install, "is_windows", lambda: False)
    monkeypatch.setattr(install, "run_command", lambda *_, **__: _completed(0, str(site_packages)))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return isaacsim_path, prebundle, site_packages


def test_repoint_replaces_every_listed_package_in_every_prebundle(fake_isaac_sim):
    """Each listed package that exists in the environment becomes a symlink in every prebundle directory."""
    isaacsim_path, prebundle, site_packages = fake_isaac_sim
    other_prebundle = isaacsim_path / "exts" / "other.ext" / "pip_prebundle"
    packages = [pkg for pkg in install._PREBUNDLE_REPOINT_PACKAGES if pkg != "nvidia"]
    for root in (prebundle, other_prebundle):
        for pkg in packages:
            (root / pkg).mkdir(parents=True)
    for pkg in packages:
        (site_packages / pkg).mkdir()
    (prebundle / "unrelated").mkdir()

    install._repoint_prebundle_packages()

    for root in (prebundle, other_prebundle):
        for pkg in packages:
            assert (root / pkg).is_symlink() and (root / pkg).resolve() == (site_packages / pkg).resolve()
    assert (prebundle / "unrelated").is_dir() and not (prebundle / "unrelated").is_symlink()


@pytest.mark.parametrize("has_cudnn", [True, False])
def test_repoint_nvidia_namespace_only_with_cuda_subpackages(fake_isaac_sim, has_cudnn):
    """The ``nvidia`` namespace holds torch's CUDA libraries, so a stripped copy must not replace it."""
    _, prebundle, site_packages = fake_isaac_sim
    (prebundle / "nvidia").mkdir()
    (site_packages / "nvidia" / ("cudnn" if has_cudnn else "srl")).mkdir(parents=True)

    install._repoint_prebundle_packages()

    assert (prebundle / "nvidia").is_symlink() is has_cudnn


@pytest.mark.parametrize("stale", [False, True], ids=["already-correct", "stale-target"])
def test_repoint_updates_existing_symlinks(fake_isaac_sim, tmp_path, stale):
    """Repointing is idempotent and moves symlinks left by a previous environment."""
    _, prebundle, site_packages = fake_isaac_sim
    (site_packages / "torch").mkdir()
    target = tmp_path / "old_env" / "torch" if stale else site_packages / "torch"
    target.mkdir(parents=True, exist_ok=True)
    (prebundle / "torch").symlink_to(target)

    install._repoint_prebundle_packages()

    assert (prebundle / "torch").resolve() == (site_packages / "torch").resolve()


def test_repoint_handles_expanded_extra_bundles(fake_isaac_sim):
    """Wheel trees expanded below ``pip_prebundle`` for extras are repointed too."""
    _, prebundle, site_packages = fake_isaac_sim
    shared_init = prebundle / "newton" / "legacy" / "__init__.py"
    shared_init.parent.mkdir(parents=True)
    shared_init.touch()
    bundled_newton = prebundle / "newton[sim]" / "newton-wheel" / "newton"
    (bundled_newton / "legacy").mkdir(parents=True)
    (bundled_newton / "legacy" / "__init__.py").symlink_to(shared_init)
    (site_packages / "newton").mkdir()

    install._repoint_prebundle_packages()

    assert (prebundle / "newton").resolve() == (site_packages / "newton").resolve()
    assert bundled_newton.is_symlink() and bundled_newton.resolve() == (site_packages / "newton").resolve()


def test_repoint_copies_on_windows(fake_isaac_sim, monkeypatch):
    """Windows has no POSIX symlinks, so the environment's package is copied into the prebundle."""
    _, prebundle, site_packages = fake_isaac_sim
    (prebundle / "torch").mkdir()
    (site_packages / "torch").mkdir()
    (site_packages / "torch" / "version.py").write_text("__version__ = '2.10.0'")
    monkeypatch.setattr(install, "is_windows", lambda: True)

    install._repoint_prebundle_packages()

    assert (prebundle / "torch").is_dir() and not (prebundle / "torch").is_symlink()
    assert (prebundle / "torch" / "version.py").exists()


def test_repoint_continues_after_an_oserror(fake_isaac_sim, monkeypatch):
    """A package that cannot be repointed is skipped with a warning; the others are still handled."""
    _, prebundle, site_packages = fake_isaac_sim
    for pkg in ("torch", "torchvision"):
        (prebundle / pkg).mkdir()
        (site_packages / pkg).mkdir()
    original_symlink_to = Path.symlink_to

    def _fail_first(self, target, **kwargs):
        if self.name == "torch":
            raise OSError("Permission denied")
        original_symlink_to(self, target, **kwargs)

    monkeypatch.setattr(Path, "symlink_to", _fail_first)

    install._repoint_prebundle_packages()

    assert not (prebundle / "torch").exists()
    assert (prebundle / "torchvision").is_symlink()


def test_repoint_fails_loudly_when_prebundled_torch_survives(fake_isaac_sim, monkeypatch):
    """A real prebundled torch left behind would shadow the pip torch on launch paths that skip isaaclab."""
    _, prebundle, site_packages = fake_isaac_sim
    (prebundle / "torch").mkdir()
    (site_packages / "torch").mkdir()
    monkeypatch.setattr(install, "_force_remove", lambda _path: None)

    with pytest.raises(RuntimeError, match="neutralize"):
        install._repoint_prebundle_packages()


@pytest.mark.parametrize("setup", ["no-isaac-sim", "no-prebundle", "probe-fails"])
def test_repoint_is_a_no_op_without_a_target(fake_isaac_sim, monkeypatch, setup):
    """Without Isaac Sim, prebundles, or a resolvable site-packages nothing is touched."""
    _, prebundle, site_packages = fake_isaac_sim
    (site_packages / "torch").mkdir()
    if setup == "no-isaac-sim":
        monkeypatch.setattr(install, "extract_isaacsim_path", lambda **_: None)
    elif setup == "no-prebundle":
        shutil.rmtree(prebundle)
    else:
        monkeypatch.setattr(install, "run_command", lambda *_, **__: _completed(1))
    if prebundle.exists():
        (prebundle / "torch").mkdir()

    install._repoint_prebundle_packages()

    assert not prebundle.exists() or not (prebundle / "torch").is_symlink()


def test_no_shadowing_prebundled_torch_in_isaac_sim():
    """A real prebundled torch must not shadow the pip-installed torch (regression: nvbugs 6343978).

    Launch paths that do not import :mod:`isaaclab` bypass the ``sys.path`` deprioritization and would
    import Isaac Sim's broken bundled copy, crashing with ``undefined symbol: ncclDevCommCreate``.
    """
    isaacsim_path = install.extract_isaacsim_path(required=False)
    if isaacsim_path is None or not isaacsim_path.exists():
        pytest.skip("Isaac Sim installation not found")

    shadowing = [p for p in isaacsim_path.rglob("pip_prebundle/torch") if p.is_dir() and not p.is_symlink()]
    assert not shadowing, "prebundled torch directories must be removed or repointed:\n  " + "\n  ".join(
        map(str, shadowing)
    )


# ---------------------------------------------------------------------------
# Prebundle symlink integrity (regression guard for nvbugs 6343978)
# ---------------------------------------------------------------------------


@pytest.fixture
def prebundle_farm(monkeypatch, tmp_path):
    """Two prebundles sharing ``packaging`` through a per-file symlink farm, as Isaac Sim ships them."""
    core = tmp_path / "exts" / "omni.isaac.core_archive" / "pip_prebundle"
    (core / "packaging").mkdir(parents=True)
    (core / "packaging" / "__init__.py").touch()
    services = tmp_path / "extscache" / "omni.services.pip_archive" / "pip_prebundle"
    (services / "packaging").mkdir(parents=True)
    (services / "packaging" / "__init__.py").symlink_to(core / "packaging" / "__init__.py")
    monkeypatch.setattr(install, "_discover_prebundle_dirs", lambda: {core, services})
    return core, services


def test_dangling_prebundle_symlinks_tolerate_preexisting_and_routine_changes(prebundle_farm, monkeypatch):
    """Only links that break after the baseline count; routine non-package dangles merely warn."""
    core, services = prebundle_farm
    (services / "stale.py").symlink_to(core / "does-not-exist.py")
    warnings = []
    monkeypatch.setattr(install, "print_warning", warnings.append)

    before = install._find_dangling_prebundle_symlinks()
    assert before == {services / "stale.py"}
    # deleting an unshared file is a routine pip replacement
    (core / "six.py").touch()
    (core / "six.py").unlink()
    install._assert_no_new_dangling_prebundle_symlinks(before)
    assert warnings == []

    # dangling files Python never imports at startup warn without failing
    (services / "WHEEL").symlink_to(core / "gone-WHEEL")
    install._assert_no_new_dangling_prebundle_symlinks(before)
    assert len(warnings) == 1


def test_dangling_package_init_fails_the_install(prebundle_farm):
    """Deleting a shared package breaks every extension linking into it and must fail loudly."""
    core, services = prebundle_farm
    before = install._find_dangling_prebundle_symlinks()
    shutil.rmtree(core / "packaging")

    with pytest.raises(RuntimeError, match="dangling symlink") as excinfo:
        install._assert_no_new_dangling_prebundle_symlinks(before)
    assert str(services / "packaging" / "__init__.py") in str(excinfo.value)
