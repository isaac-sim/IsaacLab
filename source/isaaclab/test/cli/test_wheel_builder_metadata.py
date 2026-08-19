# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for wheel-builder package metadata generated from the root pyproject."""

from __future__ import annotations

import hashlib
import http.client
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import tomllib

from tools.ci_wheelhouse import (
    build_pip_download_command,
    build_wheelhouse,
    inventory_wheel,
    load_profile,
    select_locked_wheels,
    verify_wheelhouse,
    wheel_is_compatible,
)
from tools.ci_wheelhouse import builder as ci_wheelhouse_builder
from tools.ci_wheelhouse.verify_installed import verify_installed

pytestmark = pytest.mark.unit


def _repo_root() -> Path:
    """Find the Isaac Lab repository root from this test file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source").is_dir():
            return parent
    raise RuntimeError("Could not find Isaac Lab repository root.")


def _root_rsl_rl_pin() -> str:
    """Return the ``rsl-rl-lib`` pin declared by the root ``pyproject.toml`` core deps."""
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    for dependency in data["project"]["dependencies"]:
        if dependency.startswith("rsl-rl-lib=="):
            return dependency
    raise AssertionError("Could not find rsl-rl-lib pin in the root pyproject.toml")


def _generate_wheel_pyproject(tmp_path: Path) -> dict:
    """Run ``gen_pyproject.py`` against the root pyproject and return the parsed result."""
    repo_root = _repo_root()
    output = tmp_path / "pyproject.toml"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/wheel_builder/gen_pyproject.py"),
            str(repo_root / "pyproject.toml"),
            str(output),
            "3.0.0",
        ],
        check=True,
    )
    with output.open("rb") as f:
        return tomllib.load(f)


def _generate_uv_overrides(tmp_path: Path) -> list[str]:
    """Run ``gen_uv_overrides.py`` against the root pyproject and return its requirements."""
    repo_root = _repo_root()
    output = tmp_path / "uv-overrides.txt"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/wheel_builder/gen_uv_overrides.py"),
            str(repo_root / "pyproject.toml"),
            str(output),
        ],
        check=True,
    )
    return output.read_text(encoding="utf-8").splitlines()


def _write_test_wheel(path: Path, name: str, version: str) -> str:
    """Write a tiny valid wheel archive and return its SHA-256 digest."""
    distribution = name.replace("-", "_")
    dist_info = f"{distribution}-{version}.dist-info"
    python_tag, abi_tag, platform_tag = path.stem.rsplit("-", 3)[1:]
    members = {
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n",
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: Isaac Lab test\n"
            "Root-Is-Purelib: true\n"
            f"Tag: {python_tag}-{abi_tag}-{platform_tag}\n"
        ),
        f"{dist_info}/RECORD": (f"{dist_info}/METADATA,,\n{dist_info}/WHEEL,,\n{dist_info}/RECORD,,\n"),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for member_name, contents in members.items():
            member = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, contents)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_test_profiles(path: Path) -> None:
    """Write inherited x86_64 and aarch64 test profiles."""
    path.write_text(
        "\n".join(
            [
                "schema = 1",
                "",
                "[root_groups]",
                'common = ["root-package"]',
                "",
                "[profiles.base]",
                'python_tag = "cp312"',
                'python_version = "3.12"',
                'architecture = "x86_64"',
                'platforms = ["manylinux_2_17_{architecture}"]',
                'abis = ["{python_tag}", "abi3", "none"]',
                'ci_root_groups = ["common"]',
                'lock_roots = ["root-package"]',
                "",
                "[profiles.arm]",
                'extends = "base"',
                'architecture = "aarch64"',
                "",
                "[profiles.roots-only]",
                'extends = "base"',
                "mirror_lock = false",
                'ci_roots = ["unlocked-root>=1"]',
                'ci_roots_no_deps = ["standalone-root=={python_version}"]',
                "",
                "[profiles.roots-only-child]",
                'extends = "roots-only"',
                "",
                "[profiles.hybrid]",
                'extends = "base"',
                'ci_roots = ["unlocked-root>=2"]',
                "",
                "[profiles.excluding]",
                'extends = "base"',
                'exclude_package_prefixes = ["Excluded_Pkg", "IsaacSim_{python_tag}"]',
                "",
                "[profiles.excluding-child]",
                'extends = "excluding"',
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _locked_wheel_package(
    name: str,
    version: str,
    wheel_path: Path,
    digest: str,
    *,
    resolution_markers: list[str] | None = None,
) -> str:
    """Return a registry package table for a synthetic uv lock."""
    lines = [
        "[[package]]",
        f"name = {json.dumps(name)}",
        f"version = {json.dumps(version)}",
        'source = { registry = "https://example.invalid/simple" }',
    ]
    if resolution_markers is not None:
        lines.append(f"resolution-markers = {json.dumps(resolution_markers)}")
    lines.extend(
        [
            "wheels = [",
            (f"    {{ url = {json.dumps(wheel_path.resolve().as_uri())}, hash = {json.dumps(f'sha256:{digest}')} }},"),
            "]",
            "",
        ]
    )
    return "\n".join(lines)


def _locked_package_with_wheels(name: str, version: str, wheels: list[tuple[Path, str]]) -> str:
    """Return a synthetic registry lock package with wheel alternatives."""
    wheel_entries = [
        f"    {{ url = {json.dumps(path.resolve().as_uri())}, hash = {json.dumps(f'sha256:{digest}')} }},"
        for path, digest in wheels
    ]
    return "\n".join(
        [
            "[[package]]",
            f"name = {json.dumps(name)}",
            f"version = {json.dumps(version)}",
            'source = { registry = "https://example.invalid/simple" }',
            "wheels = [",
            *wheel_entries,
            "]",
            "",
        ]
    )


def test_ci_wheelhouse_inventory_ignores_nested_vendored_metadata(tmp_path):
    """Only the wheel distribution's top-level METADATA identifies the artifact."""
    wheel_path = tmp_path / "setuptools-84.0.0-py3-none-any.whl"
    _write_test_wheel(wheel_path, "setuptools", "84.0.0")
    nested_metadata = zipfile.ZipInfo(
        "setuptools/_vendor/autocommand-2.2.2.dist-info/METADATA",
        date_time=(1980, 1, 1, 0, 0, 0),
    )
    with zipfile.ZipFile(wheel_path, "a") as archive:
        archive.writestr(
            nested_metadata,
            "Metadata-Version: 2.1\nName: autocommand\nVersion: 2.2.2\n\n",
        )

    inventory = inventory_wheel(wheel_path)

    assert inventory["name"] == "setuptools"
    assert inventory["version"] == "84.0.0"

    second_top_level_metadata = zipfile.ZipInfo(
        "other_package-1.0.dist-info/METADATA",
        date_time=(1980, 1, 1, 0, 0, 0),
    )
    with zipfile.ZipFile(wheel_path, "a") as archive:
        archive.writestr(
            second_top_level_metadata,
            "Metadata-Version: 2.1\nName: other-package\nVersion: 1.0\n\n",
        )
    with pytest.raises(ValueError, match="contains 2 top-level .dist-info/METADATA files"):
        inventory_wheel(wheel_path)

    no_metadata_wheel = tmp_path / "missing_metadata-1.0-py3-none-any.whl"
    with zipfile.ZipFile(no_metadata_wheel, "w"):
        pass
    with pytest.raises(ValueError, match="contains 0 top-level .dist-info/METADATA files"):
        inventory_wheel(no_metadata_wheel)


def test_wheel_builder_drops_workspace_members(tmp_path):
    """The generated wheel metadata must not depend on the bundled ``isaaclab*`` packages."""
    generated = _generate_wheel_pyproject(tmp_path)
    dependencies = generated["project"]["dependencies"]

    assert not [dep for dep in dependencies if dep.lower().startswith("isaaclab")]


def test_wheel_builder_includes_isaacsim_extra(tmp_path):
    """The ``isaacsim`` extra must ship in the generated wheel metadata."""
    generated = _generate_wheel_pyproject(tmp_path)
    optional_dependencies = generated["project"]["optional-dependencies"]

    assert "isaacsim" in optional_dependencies
    assert any(dep.startswith("isaacsim[") for dep in optional_dependencies["isaacsim"])


def test_wheel_builder_expands_all_extra_into_concrete_requirements(tmp_path):
    """``isaaclab[all]`` must ship the aggregated requirements, not a self-reference.

    At the root, ``all`` is the self-reference ``isaaclab-dev[...]``. The generator
    inlines it, so the published wheel carries the concrete third-party requirements
    for every backend, RL library, and visualizer.
    """
    generated = _generate_wheel_pyproject(tmp_path)
    optional_dependencies = generated["project"]["optional-dependencies"]
    all_extra = optional_dependencies["all"]

    assert not any(dep.lower().startswith("isaaclab") for dep in all_extra)
    # Sampled across what ``all`` aggregates: Isaac Sim, both OV backends, the RL
    # libraries, and the visualizers.
    for prefix in ("isaacsim[", "ovphysx", "ovrtx", "ovstage", "stable-baselines3", "skrl", "viser", "rerun-sdk"):
        assert any(dep.startswith(prefix) for dep in all_extra), f"'{prefix}' missing from the 'all' extra"
    # The specialized extras and the developer tooling stay opt-in by name.
    for prefix in ("ray", "robomimic", "isaacteleop", "pytetwild", "moviepy", "leapp", "pytest"):
        assert not any(dep.startswith(prefix) for dep in all_extra), f"'{prefix}' must not be in the 'all' extra"


def test_wheel_builder_rsl_rl_pin_matches_root_pyproject(tmp_path):
    """The bundled wheel metadata must install the RSL-RL version declared at the root."""
    expected_pin = _root_rsl_rl_pin()
    generated = _generate_wheel_pyproject(tmp_path)

    # RSL-RL is a core dependency (default training library) and also exposed as an extra.
    core_pins = [dep for dep in generated["project"]["dependencies"] if dep.startswith("rsl-rl-lib==")]
    assert core_pins == [expected_pin]

    optional_dependencies = generated["project"]["optional-dependencies"]
    # RSL-RL is also exposed through its own ``rsl-rl`` extra.
    rsl_rl_pins = [dep for dep in optional_dependencies["rsl-rl"] if dep.startswith("rsl-rl-lib==")]
    assert rsl_rl_pins == [expected_pin]


def test_wheel_builder_keeps_tetrahedralization_explicit(tmp_path):
    """The generated wheel must expose PyTetWild only through its explicit extra."""
    generated = _generate_wheel_pyproject(tmp_path)
    project = generated["project"]
    optional_dependencies = project["optional-dependencies"]

    assert not any(dep.startswith("pytetwild") for dep in project["dependencies"])
    assert optional_dependencies["tetrahedralization"] == ["pytetwild[all]>=0.3.0,<0.4"]
    for name, deps in optional_dependencies.items():
        if name == "tetrahedralization":
            continue
        assert not any(dep.startswith("pytetwild") for dep in deps)


def test_wheel_builder_uv_overrides_match_root_pyproject(tmp_path):
    """The wheel resolver override file must mirror the root uv overrides exactly."""
    with (_repo_root() / "pyproject.toml").open("rb") as f:
        root = tomllib.load(f)

    generated_overrides = _generate_uv_overrides(tmp_path)
    published_overrides = (
        (_repo_root() / "tools" / "wheel_builder" / "uv-overrides.txt").read_text(encoding="utf-8").splitlines()
    )
    install_ci_overrides = (
        (_repo_root() / "source" / "isaaclab" / "test" / "install_ci" / "uv_pip" / "uv-overrides.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert generated_overrides == root["tool"]["uv"]["override-dependencies"]
    assert published_overrides == generated_overrides
    assert install_ci_overrides == generated_overrides


def test_wheel_builder_uv_overrides_relax_isaacsim_exact_pins(tmp_path):
    """The wheel resolver must relax Isaac Sim 6.0's exact pins so the extras co-resolve."""
    overrides = _generate_uv_overrides(tmp_path)

    for spec in ("typing-extensions>=4.15.0", "websockets>=14.0,<17.0.0", "coverage>=7.6.1"):
        assert spec in overrides


def test_ci_wheelhouse_profile_loading_and_inheritance(tmp_path):
    """Profiles inherit target settings, expand roots, and substitute architecture/version fields."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)

    base = load_profile(profiles_path, "base")
    arm = load_profile(profiles_path, "arm")
    excluding_child = load_profile(profiles_path, "excluding-child")
    roots_only_child = load_profile(profiles_path, "roots-only-child")

    assert base.architecture == "x86_64"
    assert base.platforms == ("manylinux_2_17_x86_64",)
    assert base.ci_roots == ("root-package",)
    assert base.ci_roots_no_deps == ()
    assert base.exclude_package_prefixes == ()
    assert base.mirror_lock is True
    assert arm.architecture == "aarch64"
    assert arm.platforms == ("manylinux_2_17_aarch64",)
    assert arm.abis == ("cp312", "abi3", "none")
    assert excluding_child.exclude_package_prefixes == ("excluded-pkg", "isaacsim-cp312")
    assert roots_only_child.ci_roots_no_deps == ("standalone-root==3.12",)

    checked_in_profiles = _repo_root() / ".github" / "ci-wheelhouse" / "profiles.toml"
    hosted = load_profile(checked_in_profiles, "hosted-cp312-x86_64")
    assert hosted.mirror_lock is False
    for root in (
        "uv==0.12.5",
        "build==1.5.0",
        "wheel==0.48.0",
        "poetry-core==2.4.1",
        "hatchling==1.32.0",
        "pip-licenses==5.5.5",
        "pipdeptree==4.2.1",
        "pytest-timeout==2.4.0",
        "pre-commit==4.6.2",
    ):
        assert root in hosted.ci_roots
    assert not {"leapp", "ovphysx", "pytetwild[all]>=0.3,<0.4"} & set(hosted.ci_roots)

    runtime = load_profile(checked_in_profiles, "runtime-cp312-x86_64")
    assert runtime.ci_roots_no_deps == ("leapp",)
    assert runtime.ci_roots == ("uv==0.12.5", "jsonschema", "pytetwild[all]>=0.3,<0.4", "decorator<5")

    isaacsim = load_profile(checked_in_profiles, "isaacsim-cp312-x86_64")
    assert isaacsim.mirror_lock is True
    assert isaacsim.exclude_package_prefixes == ("isaacsim",)
    assert "leapp" in isaacsim.ci_roots
    assert "leapp" in isaacsim.lock_roots
    with pytest.raises(ValueError, match="requires --base_version"):
        load_profile(checked_in_profiles, "compatibility-cp312-x86_64")
    compatibility = load_profile(
        checked_in_profiles,
        "compatibility-cp312-x86_64",
        base_version="5.0.0",
    )
    assert "isaacsim[all,extscache]==5.0.0" in compatibility.ci_roots
    assert compatibility.exclude_package_prefixes == ("isaacsim",)
    assert compatibility.mirror_lock is True
    assert load_profile(checked_in_profiles, "hosted-full-cp310-x86_64").mirror_lock is False
    assert load_profile(checked_in_profiles, "compatibility-cp310-x86_64", base_version="5.0.0").mirror_lock is False
    assert load_profile(checked_in_profiles, "compatibility-cp311-x86_64", base_version="5.0.0").mirror_lock is False

    command = build_pip_download_command(arm, "/target/python", tmp_path / "wheelhouse", ["hatchling"])
    assert command[:4] == ["/target/python", "-m", "pip", "download"]
    assert command[command.index("--python-version") + 1] == "3.12"
    assert command[command.index("--platform") + 1] == "manylinux_2_17_aarch64"
    assert command[-1] == "hatchling"

    no_deps_command = build_pip_download_command(
        roots_only_child,
        "/target/python",
        tmp_path / "wheelhouse",
        roots_only_child.ci_roots_no_deps,
        no_deps=True,
    )
    assert "--no-deps" in no_deps_command
    assert no_deps_command[-1] == "standalone-root==3.12"


@pytest.mark.parametrize(
    ("filename", "architecture", "expected"),
    [
        ("pure_pkg-1.0-py3-none-any.whl", "x86_64", True),
        ("native_pkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl", "x86_64", True),
        ("limited_pkg-1.0-cp38-abi3-manylinux2014_x86_64.whl", "x86_64", True),
        ("arm_pkg-1.0-cp312-cp312-manylinux_2_17_aarch64.whl", "aarch64", True),
        ("arm_pkg-1.0-cp312-cp312-manylinux_2_17_aarch64.whl", "x86_64", False),
        ("x86_pkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl", "aarch64", False),
        ("future_pkg-1.0-cp313-abi3-manylinux_2_17_x86_64.whl", "x86_64", False),
        ("older_pkg-1.0-cp311-cp311-manylinux_2_17_x86_64.whl", "x86_64", False),
        ("musl_pkg-1.0-cp312-cp312-musllinux_1_2_x86_64.whl", "x86_64", False),
    ],
)
def test_ci_wheelhouse_wheel_tag_compatibility(filename, architecture, expected):
    """Wheel filtering accepts only compatible Python, ABI, manylinux, and architecture tags."""
    assert wheel_is_compatible(filename, "cp312", architecture) is expected


def test_ci_wheelhouse_rejects_unsupported_manylinux_floor():
    """Every glibc floor through the target maximum is accepted, but future floors are not."""
    target_platforms = ("manylinux_2_35_x86_64", "manylinux_2_17_x86_64")
    for platform_tag in (
        "manylinux_2_31_x86_64",
        "manylinux_2_27_x86_64",
        "manylinux_2_18_x86_64",
        "manylinux2014_x86_64",
        "manylinux2010_x86_64",
    ):
        filename = f"compatible_pkg-1.0-cp312-cp312-{platform_tag}.whl"
        assert wheel_is_compatible(filename, "cp312", "x86_64", target_platforms) is True

    future = "future_pkg-1.0-cp312-cp312-manylinux_2_99_x86_64.whl"
    assert wheel_is_compatible(future, "cp312", "x86_64", target_platforms) is False
    assert wheel_is_compatible(future, "cp312", "x86_64", ("manylinux_2_99_x86_64",)) is True


def test_ci_wheelhouse_resolution_markers_select_target_architecture(tmp_path):
    """Package-level Linux architecture markers select only the matching lock entry."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    x86_profile = load_profile(profiles_path, "base")
    arm_profile = load_profile(profiles_path, "arm")
    x86_wheel = tmp_path / "marker_pkg-1.0-py3-none-any.whl"
    arm_wheel = tmp_path / "marker_pkg-2.0-py3-none-any.whl"
    x86_digest = _write_test_wheel(x86_wheel, "marker-pkg", "1.0")
    arm_digest = _write_test_wheel(arm_wheel, "marker-pkg", "2.0")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n"
        'requires-python = "==3.12.*"\n\n'
        + _locked_wheel_package(
            "marker-pkg",
            "1.0",
            x86_wheel,
            x86_digest,
            resolution_markers=["platform_machine == 'x86_64' and sys_platform == 'linux'"],
        )
        + _locked_wheel_package(
            "marker-pkg",
            "2.0",
            arm_wheel,
            arm_digest,
            resolution_markers=["platform_machine == 'aarch64' and sys_platform == 'linux'"],
        ),
        encoding="utf-8",
    )

    x86_selection = select_locked_wheels(lock_path, x86_profile)
    arm_selection = select_locked_wheels(lock_path, arm_profile)
    assert [(wheel.package_name, wheel.package_version) for wheel in x86_selection.wheels] == [("marker-pkg", "1.0")]
    assert [(wheel.package_name, wheel.package_version) for wheel in arm_selection.wheels] == [("marker-pkg", "2.0")]


def test_ci_wheelhouse_rejects_mirrored_python_incompatible_lock(tmp_path):
    """A mirror profile cannot select from a lock that excludes its Python."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "base", python_tag="cp311")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text('version = 1\nrequires-python = "==3.12.*"\npackage = []\n', encoding="utf-8")

    with pytest.raises(ValueError, match="requires-python.*excludes profile Python 3.11.*mirror_lock=false"):
        select_locked_wheels(lock_path, profile)


def test_ci_wheelhouse_selects_one_preferred_wheel(tmp_path):
    """Selection prefers exact CPython, then the highest supported manylinux floor."""
    profiles_path = _repo_root() / ".github" / "ci-wheelhouse" / "profiles.toml"
    profile = load_profile(profiles_path, "hosted-full-cp312-x86_64")
    alternatives = [
        tmp_path / "preference_pkg-1.0-py3-none-any.whl",
        tmp_path / "preference_pkg-1.0-cp38-abi3-manylinux_2_35_x86_64.whl",
        tmp_path / "preference_pkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl",
        tmp_path / "preference_pkg-1.0-0-cp312-cp312-manylinux_2_28_x86_64.whl",
        tmp_path / "preference_pkg-1.0-1-cp312-cp312-manylinux_2_28_x86_64.whl",
        tmp_path / "preference_pkg-1.0-cp312-cp312-manylinux_2_99_x86_64.whl",
    ]
    wheel_digests = [(path, _write_test_wheel(path, "preference-pkg", "1.0")) for path in alternatives]
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        'version = 1\nrequires-python = "==3.12.*"\n\n'
        + _locked_package_with_wheels("preference-pkg", "1.0", wheel_digests),
        encoding="utf-8",
    )

    selection = select_locked_wheels(lock_path, profile)

    assert [wheel.filename for wheel in selection.wheels] == [
        "preference_pkg-1.0-1-cp312-cp312-manylinux_2_28_x86_64.whl"
    ]


def test_ci_wheelhouse_full_x86_selection_includes_open3d_floor():
    """The real full x86 lock selection accepts Open3D's manylinux 2.31 wheel."""
    repo_root = _repo_root()
    profile = load_profile(
        repo_root / ".github" / "ci-wheelhouse" / "profiles.toml",
        "hosted-full-cp312-x86_64",
    )

    selection = select_locked_wheels(repo_root / "uv.lock", profile)

    assert selection.errors == ()
    assert any(wheel.filename == "open3d-0.19.0-cp312-cp312-manylinux_2_31_x86_64.whl" for wheel in selection.wheels)
    assert not any(item["name"] == "open3d" for item in selection.exclusions)


def test_ci_wheelhouse_selects_lock_wheels_and_documents_exclusions(tmp_path):
    """Synthetic lock selection keeps compatible URLs and documents unsupported source kinds."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "base")

    pure_wheel = tmp_path / "pure_pkg-1.0-py3-none-any.whl"
    native_wheel = tmp_path / "native_pkg-2.0-cp312-cp312-manylinux_2_17_x86_64.whl"
    abi3_wheel = tmp_path / "abi_pkg-3.0-cp38-abi3-manylinux2014_x86_64.whl"
    arm_wheel = tmp_path / "arm_pkg-4.0-cp312-cp312-manylinux_2_17_aarch64.whl"
    wheel_digests = {
        pure_wheel: _write_test_wheel(pure_wheel, "pure-pkg", "1.0"),
        native_wheel: _write_test_wheel(native_wheel, "native-pkg", "2.0"),
        abi3_wheel: _write_test_wheel(abi3_wheel, "abi-pkg", "3.0"),
        arm_wheel: _write_test_wheel(arm_wheel, "arm-pkg", "4.0"),
    }
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n\n"
        + "".join(
            _locked_wheel_package(
                wheel_path.name.split("-")[0].replace("_", "-"),
                version,
                wheel_path,
                wheel_digests[wheel_path],
            )
            for wheel_path, version in (
                (pure_wheel, "1.0"),
                (native_wheel, "2.0"),
                (abi3_wheel, "3.0"),
                (arm_wheel, "4.0"),
            )
        )
        + "\n".join(
            [
                "[[package]]",
                'name = "git-package"',
                'version = "1.0"',
                'source = { git = "https://example.invalid/repository.git#deadbeef" }',
                "",
                "[[package]]",
                'name = "editable-package"',
                'version = "1.0"',
                'source = { editable = "source/editable_package" }',
                "",
                "[[package]]",
                'name = "source-only"',
                'version = "1.0"',
                'source = { registry = "https://example.invalid/simple" }',
                (f'sdist = {{ url = "https://example.invalid/source-only-1.0.tar.gz", hash = "sha256:{"0" * 64}" }}'),
                "",
            ]
        ),
        encoding="utf-8",
    )

    selection = select_locked_wheels(lock_path, profile)

    assert [(wheel.package_name, wheel.package_version) for wheel in selection.wheels] == [
        ("abi-pkg", "3.0"),
        ("native-pkg", "2.0"),
        ("pure-pkg", "1.0"),
    ]
    assert not selection.errors
    exclusions = {entry["name"]: entry["reason"] for entry in selection.exclusions}
    assert exclusions == {
        "arm-pkg": "incompatible-wheel-tags",
        "editable-package": "editable-source",
        "git-package": "git-source",
        "source-only": "sdist-only",
    }


def test_ci_wheelhouse_profile_exclusions_skip_normalized_prefixes_and_remain_complete(tmp_path):
    """Exact and prefix exclusions are intentional, normalized, and do not make builds partial."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "excluding-child")
    retained_wheel = tmp_path / "retained_package-1.0-py3-none-any.whl"
    excluded_exact_wheel = tmp_path / "excluded_pkg-1.0-py3-none-any.whl"
    excluded_prefix_wheel = tmp_path / "excluded_pkg_plugin-1.0-py3-none-any.whl"
    excluded_template_wheel = tmp_path / "isaacsim_cp312_extcache-1.0-py3-none-any.whl"
    wheels = (
        ("retained-package", retained_wheel),
        ("Excluded.Pkg", excluded_exact_wheel),
        ("excluded_pkg_plugin", excluded_prefix_wheel),
        ("IsaacSim_CP312_ExtCache", excluded_template_wheel),
    )
    wheel_digests = {
        wheel_path: _write_test_wheel(wheel_path, package_name, "1.0") for package_name, wheel_path in wheels
    }
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n\n"
        + "".join(
            _locked_wheel_package(package_name, "1.0", wheel_path, wheel_digests[wheel_path])
            for package_name, wheel_path in wheels
        ),
        encoding="utf-8",
    )
    for excluded_wheel in (excluded_exact_wheel, excluded_prefix_wheel, excluded_template_wheel):
        excluded_wheel.unlink()

    output_dir = tmp_path / "output"
    manifest = build_wheelhouse(
        lock_path,
        profiles_path,
        profile,
        output_dir,
        include_ci_roots=False,
        attempts=1,
        backoff_seconds=0,
        workers=1,
    )

    assert manifest["complete"] is True
    assert manifest["errors"] == []
    assert manifest["profile"]["exclude_package_prefixes"] == ["excluded-pkg", "isaacsim-cp312"]
    assert [record["filename"] for record in manifest["files"]] == [retained_wheel.name]
    assert {(item["name"], item["reason"]) for item in manifest["exclusions"]} == {
        ("excluded-pkg", "profile-excluded"),
        ("excluded-pkg-plugin", "profile-excluded"),
        ("isaacsim-cp312-extcache", "profile-excluded"),
    }
    assert verify_wheelhouse(output_dir, require_complete=True) == []

    manifest_path = output_dir / "manifest.json"
    altered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    altered_manifest["profile"]["exclude_package_prefixes"] = []
    manifest_path.write_text(json.dumps(altered_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_wheelhouse(output_dir) == ["profile input hash mismatch"]


def test_ci_wheelhouse_roots_only_profile_skips_lock_mirroring(tmp_path, monkeypatch):
    """A roots-only profile sends every CI root to pip and never selects lock wheels."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "roots-only")
    locked_wheel = tmp_path / "root_package-9.9-py3-none-any.whl"
    digest = _write_test_wheel(locked_wheel, "root-package", "9.9")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n\n" + _locked_wheel_package("root-package", "9.9", locked_wheel, digest),
        encoding="utf-8",
    )
    pip_commands = []

    def reject_lock_selection(*_args, **_kwargs):
        pytest.fail("mirror_lock=false must not select registry lock wheels")

    def fake_pip_download(command, **_kwargs):
        pip_commands.append(command)
        destination = Path(command[command.index("--dest") + 1])
        constraint_path = Path(command[command.index("--constraint") + 1])
        assert constraint_path.read_text(encoding="utf-8") == "root-package==9.9\n"
        if "--no-deps" in command:
            _write_test_wheel(destination / "standalone_root-3.12-py3-none-any.whl", "standalone-root", "3.12")
            return
        _write_test_wheel(destination / "unlocked_root-1.0-py3-none-any.whl", "unlocked-root", "1.0")
        _write_test_wheel(destination / "root_package-1.0-py3-none-any.whl", "root-package", "1.0")
        _write_test_wheel(destination / "root_closure-2.0-py3-none-any.whl", "root-closure", "2.0")

    monkeypatch.setattr(ci_wheelhouse_builder, "select_locked_wheels", reject_lock_selection)
    monkeypatch.setattr(ci_wheelhouse_builder, "_run_pip_download", fake_pip_download)
    output_dir = tmp_path / "output"
    wheelhouse_path = output_dir / "wheelhouse"
    wheelhouse_path.mkdir(parents=True)
    stale_normal_wheel = wheelhouse_path / "unlocked_root-0.9-py3-none-any.whl"
    stale_no_deps_wheel = wheelhouse_path / "standalone_root-3.11-py3-none-any.whl"
    _write_test_wheel(stale_normal_wheel, "unlocked-root", "0.9")
    _write_test_wheel(stale_no_deps_wheel, "standalone-root", "3.11")

    manifest = build_wheelhouse(
        lock_path,
        profiles_path,
        profile,
        output_dir,
        target_python="/target/python",
        attempts=1,
        backoff_seconds=0,
        workers=1,
    )

    assert manifest["complete"] is True
    assert manifest["profile"]["mirror_lock"] is False
    assert manifest["profile"]["ci_roots_no_deps"] == ["standalone-root==3.12"]
    assert manifest["completeness"]["required_lock_files"] == 0
    assert manifest["exclusions"] == []
    assert manifest["roots"] == {
        "enabled": True,
        "excluded": [],
        "locked": [],
        "constraints": {
            "count": 1,
            "sha256": hashlib.sha256(b"root-package==9.9\n").hexdigest(),
        },
        "pip": ["unlocked-root>=1", "root-package"],
        "pip_no_deps": ["standalone-root==3.12"],
    }
    assert len(pip_commands) == 2
    assert pip_commands[0][-2:] == ["unlocked-root>=1", "root-package"]
    assert "--no-deps" not in pip_commands[0]
    assert "--no-deps" in pip_commands[1]
    assert pip_commands[1][-1] == "standalone-root==3.12"
    assert {record["filename"] for record in manifest["files"]} == {
        "root_closure-2.0-py3-none-any.whl",
        "root_package-1.0-py3-none-any.whl",
        "standalone_root-3.12-py3-none-any.whl",
        "unlocked_root-1.0-py3-none-any.whl",
    }
    assert stale_normal_wheel.exists() is False
    assert stale_no_deps_wheel.exists() is False
    assert locked_wheel.name not in {record["filename"] for record in manifest["files"]}
    assert verify_wheelhouse(output_dir, lock_path=lock_path, profiles_path=profiles_path) == []

    manifest_path = output_dir / "manifest.json"
    missing_normal_root = json.loads(json.dumps(manifest))
    missing_normal_root["packages"] = [
        package for package in missing_normal_root["packages"] if package["name"] != "unlocked-root"
    ]
    manifest_path.write_text(json.dumps(missing_normal_root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert "manifest root requirement is missing package inventory: unlocked-root>=1" in verify_wheelhouse(output_dir)

    missing_normal_root["roots"]["excluded"] = ["unlocked-root>=1"]
    manifest_path.write_text(json.dumps(missing_normal_root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing_normal_error = "manifest root requirement is missing package inventory: unlocked-root>=1"
    assert missing_normal_error not in verify_wheelhouse(output_dir)

    wrong_exact_root = json.loads(json.dumps(manifest))
    for package in wrong_exact_root["packages"]:
        if package["name"] == "standalone-root":
            package["version"] = "0"
    manifest_path.write_text(json.dumps(wrong_exact_root, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing_exact_error = "manifest root requirement is missing package inventory: standalone-root==3.12"
    assert missing_exact_error in verify_wheelhouse(output_dir)

    altered_manifest = json.loads(json.dumps(manifest))
    altered_manifest["profile"]["ci_roots_no_deps"] = []
    manifest_path.write_text(json.dumps(altered_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_wheelhouse(output_dir) == ["profile input hash mismatch"]


def test_ci_wheelhouse_constraints_use_unambiguous_compatible_lock_versions(tmp_path):
    """Constraints include one target-compatible registry version and omit ambiguous names."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "roots-only")
    unique_wheel = tmp_path / "unique_pkg-1.0-py3-none-any.whl"
    duplicate_one = tmp_path / "duplicate_pkg-1.0-py3-none-any.whl"
    duplicate_two = tmp_path / "duplicate_pkg-2.0-py3-none-any.whl"
    arm_wheel = tmp_path / "arm_only-3.0-py3-none-any.whl"
    wheels = [
        ("unique-pkg", "1.0", unique_wheel, None),
        ("duplicate-pkg", "1.0", duplicate_one, None),
        ("duplicate-pkg", "2.0", duplicate_two, None),
        (
            "arm-only",
            "3.0",
            arm_wheel,
            ["platform_machine == 'aarch64' and sys_platform == 'linux'"],
        ),
    ]
    packages = []
    for name, version, wheel_path, markers in wheels:
        digest = _write_test_wheel(wheel_path, name, version)
        packages.append(
            _locked_wheel_package(
                name,
                version,
                wheel_path,
                digest,
                resolution_markers=markers,
            )
        )
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        'version = 1\nrequires-python = "==3.12.*"\n\n' + "".join(packages),
        encoding="utf-8",
    )

    assert ci_wheelhouse_builder._lock_constraints(lock_path, profile) == ("unique-pkg==1.0",)


def test_ci_wheelhouse_prunes_stale_restore_and_reuses_locked_wheel(tmp_path, monkeypatch):
    """A broad restore keeps an exact lock hit, prunes stale roots, and merges partial pip results."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "hybrid")
    locked_source = tmp_path / "root_package-1.0-py3-none-any.whl"
    locked_digest = _write_test_wheel(locked_source, "root-package", "1.0")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n\n" + _locked_wheel_package("root-package", "1.0", locked_source, locked_digest),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    wheelhouse_path = output_dir / "wheelhouse"
    wheelhouse_path.mkdir(parents=True)
    cached_locked_wheel = wheelhouse_path / locked_source.name
    cached_locked_wheel.write_bytes(locked_source.read_bytes())
    stale_wheel = wheelhouse_path / "unlocked_root-1.0-py3-none-any.whl"
    _write_test_wheel(stale_wheel, "unlocked-root", "1.0")
    staging_paths = []
    staged_digest = ""

    def reject_locked_download(*_args, **_kwargs):
        pytest.fail("an exact locked-wheel cache hit must not be downloaded again")

    def partially_failing_pip(command, **_kwargs):
        nonlocal staged_digest
        staging_path = Path(command[command.index("--dest") + 1])
        staging_paths.append(staging_path)
        assert staging_path != wheelhouse_path
        staged_wheel = staging_path / "unlocked_root-2.0-py3-none-any.whl"
        staged_digest = _write_test_wheel(staged_wheel, "unlocked-root", "2.0")
        return f"simulated pip failure in {staging_path}"

    monkeypatch.setattr(ci_wheelhouse_builder.urllib.request, "urlopen", reject_locked_download)
    monkeypatch.setattr(ci_wheelhouse_builder, "_run_pip_download", partially_failing_pip)

    manifest = build_wheelhouse(
        lock_path,
        profiles_path,
        profile,
        output_dir,
        target_python="/target/python",
        attempts=1,
        backoff_seconds=0,
        workers=1,
    )

    current_root_wheel = wheelhouse_path / "unlocked_root-2.0-py3-none-any.whl"
    assert manifest["complete"] is False
    assert manifest["completeness"]["present_lock_files"] == 1
    assert {record["filename"] for record in manifest["files"]} == {
        cached_locked_wheel.name,
        current_root_wheel.name,
    }
    assert {record["sha256"] for record in manifest["files"]} == {locked_digest, staged_digest}
    assert stale_wheel.exists() is False
    assert cached_locked_wheel.read_bytes() == locked_source.read_bytes()
    assert current_root_wheel.is_file()
    assert staging_paths and staging_paths[0].exists() is False
    assert "<ci-root-staging>" in manifest["errors"][0]["command"]
    assert "<ci-root-staging>" in manifest["errors"][0]["error"]
    assert (output_dir / "complete").exists() is False
    assert verify_wheelhouse(output_dir, require_complete=False) == []


def test_ci_wheelhouse_retries_transient_local_download(tmp_path, monkeypatch):
    """A transient exact-URL failure is retried with bounded exponential backoff."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "base")
    source_wheel = tmp_path / "retry_pkg-1.0-py3-none-any.whl"
    digest = _write_test_wheel(source_wheel, "retry-pkg", "1.0")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n\n" + _locked_wheel_package("retry-pkg", "1.0", source_wheel, digest),
        encoding="utf-8",
    )

    original_urlopen = ci_wheelhouse_builder.urllib.request.urlopen
    attempts = 0
    delays = []

    def flaky_urlopen(request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient test failure")
        return original_urlopen(request, timeout=timeout)

    monkeypatch.setattr(ci_wheelhouse_builder.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(ci_wheelhouse_builder.time, "sleep", delays.append)

    manifest = build_wheelhouse(
        lock_path,
        profiles_path,
        profile,
        tmp_path / "output",
        include_ci_roots=False,
        attempts=2,
        backoff_seconds=0.25,
        max_backoff_seconds=1.0,
        workers=1,
    )

    assert manifest["complete"] is True
    assert attempts == 2
    assert delays == [0.25]


def test_ci_wheelhouse_retries_incomplete_stream_and_prunes_parts(tmp_path, monkeypatch):
    """Incomplete HTTP streams are retried and stale partial artifacts are removed."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "base")
    source_wheel = tmp_path / "stream_pkg-1.0-py3-none-any.whl"
    digest = _write_test_wheel(source_wheel, "stream-pkg", "1.0")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n\n" + _locked_wheel_package("stream-pkg", "1.0", source_wheel, digest),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    wheelhouse_path = output_dir / "wheelhouse"
    wheelhouse_path.mkdir(parents=True)
    (wheelhouse_path / ".restored-download.part").write_bytes(b"stale")
    original_urlopen = ci_wheelhouse_builder.urllib.request.urlopen
    attempts = 0

    class IncompleteResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            raise http.client.IncompleteRead(b"partial", 100)

    def incomplete_once(request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return IncompleteResponse()
        return original_urlopen(request, timeout=timeout)

    monkeypatch.setattr(ci_wheelhouse_builder.urllib.request, "urlopen", incomplete_once)

    manifest = build_wheelhouse(
        lock_path,
        profiles_path,
        profile,
        output_dir,
        include_ci_roots=False,
        attempts=2,
        backoff_seconds=0,
        workers=1,
    )

    assert manifest["complete"] is True
    assert attempts == 2
    assert list(wheelhouse_path.glob("*.part")) == []


def test_ci_wheelhouse_manifest_is_deterministic_verified_and_cacheable(tmp_path):
    """A local locked wheel produces a stable verified manifest and is reused without its source."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "base")
    source_wheel = tmp_path / "cached_pkg-1.2.3-py3-none-any.whl"
    digest = _write_test_wheel(source_wheel, "cached-pkg", "1.2.3")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n\n" + _locked_wheel_package("cached-pkg", "1.2.3", source_wheel, digest),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    first_manifest = build_wheelhouse(
        lock_path,
        profiles_path,
        profile,
        output_dir,
        include_ci_roots=False,
        attempts=1,
        backoff_seconds=0,
        workers=1,
    )

    assert first_manifest["complete"] is True
    assert first_manifest["completeness"] == {
        "failed": 0,
        "present_lock_files": 1,
        "required_lock_files": 1,
        "status": "complete",
    }
    assert first_manifest["files"][0]["sha256"] == digest
    assert first_manifest["packages"] == [
        {
            "files": [source_wheel.name],
            "name": "cached-pkg",
            "version": "1.2.3",
        }
    ]
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == first_manifest
    assert (output_dir / "complete").read_text(encoding="utf-8") == "complete\n"
    assert verify_wheelhouse(output_dir, lock_path=lock_path, profiles_path=profiles_path) == []

    source_wheel.unlink()
    second_manifest = build_wheelhouse(
        lock_path,
        profiles_path,
        profile,
        output_dir,
        include_ci_roots=False,
        attempts=2,
        backoff_seconds=0,
        workers=1,
    )
    assert second_manifest == first_manifest

    downloaded_wheel = output_dir / "wheelhouse" / source_wheel.name
    downloaded_wheel.write_bytes(downloaded_wheel.read_bytes() + b"tampered")
    assert verify_wheelhouse(output_dir) == [f"sha256 mismatch: {source_wheel.name}"]


def test_ci_wheelhouse_verification_rejects_contradictory_complete_manifests(tmp_path):
    """Verification rejects forged completeness and current lock representation claims."""
    profiles_path = tmp_path / "profiles.toml"
    _write_test_profiles(profiles_path)
    profile = load_profile(profiles_path, "base")
    source_wheel = tmp_path / "verified_pkg-1.0-py3-none-any.whl"
    digest = _write_test_wheel(source_wheel, "verified-pkg", "1.0")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(
        "version = 1\n\n" + _locked_wheel_package("verified-pkg", "1.0", source_wheel, digest),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    manifest = build_wheelhouse(
        lock_path,
        profiles_path,
        profile,
        output_dir,
        include_ci_roots=False,
        attempts=1,
        backoff_seconds=0,
        workers=1,
    )
    manifest_path = output_dir / "manifest.json"

    def verify_mutation(mutator, expected_error):
        altered = json.loads(json.dumps(manifest))
        mutator(altered)
        manifest_path.write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        assert expected_error in verify_wheelhouse(
            output_dir,
            lock_path=lock_path,
            profiles_path=profiles_path,
        )

    verify_mutation(
        lambda value: (value["errors"].append({"error": "forged"}), value["completeness"].update(failed=1)),
        "complete manifest must not contain errors",
    )
    verify_mutation(
        lambda value: value["completeness"].update(status="partial"),
        "manifest completeness status must be 'complete'",
    )
    verify_mutation(
        lambda value: value["completeness"].pop("required_lock_files"),
        "manifest completeness required_lock_files must be a non-negative integer",
    )
    verify_mutation(
        lambda value: value["completeness"].update(present_lock_files=0),
        "complete manifest must represent every required lock file",
    )
    verify_mutation(
        lambda value: value.update(files=[]),
        "manifest files must not be empty when requirements exist",
    )
    verify_mutation(
        lambda value: value["files"][0].update(origin="ci-root"),
        f"manifest lock origin mismatch: {source_wheel.name}",
    )
    verify_mutation(
        lambda value: value["files"][0].update(sha256="0" * 64),
        f"manifest lock sha256 mismatch: {source_wheel.name}",
    )


def test_ci_wheelhouse_verify_installed_list_manifest(monkeypatch):
    """Installed versions are checked against the current list-style package inventory."""
    requested_names = []

    def installed_version(name):
        requested_names.append(name)
        return "1.2.3"

    monkeypatch.setattr("tools.ci_wheelhouse.verify_installed.importlib.metadata.version", installed_version)
    manifest = {
        "schema": 1,
        "packages": [
            {
                "files": ["demo_package-1.2.3-py3-none-any.whl"],
                "name": "demo-package",
                "version": "1.2.3",
            }
        ],
    }

    errors = verify_installed(
        manifest,
        ["Demo_Package>=1"],
        [],
        required_fallback_used=False,
    )

    assert errors == []
    assert requested_names == ["demo-package"]


def test_ci_wheelhouse_verify_installed_legacy_manifest_does_not_import_unrequested_ovphysx(monkeypatch):
    """Legacy manifests keep non-OVPhysX requests compatible without importing OVPhysX."""

    def unexpected_call(*args, **kwargs):
        raise AssertionError(f"Unexpected package lookup or import: {args}, {kwargs}")

    monkeypatch.setattr("tools.ci_wheelhouse.verify_installed.importlib.metadata.version", unexpected_call)
    monkeypatch.setattr("tools.ci_wheelhouse.verify_installed.importlib.import_module", unexpected_call)

    errors = verify_installed(
        {"ovphysx_version": "0.5.10"},
        ["pytest"],
        [],
        required_fallback_used=False,
    )

    assert errors == []


def test_ci_wheelhouse_verify_installed_honors_exclusions(monkeypatch):
    """Manifest exclusions do not trigger installed-version lookups."""

    def unexpected_lookup(name):
        raise AssertionError(f"Unexpected installed-version lookup: {name}")

    monkeypatch.setattr("tools.ci_wheelhouse.verify_installed.importlib.metadata.version", unexpected_lookup)
    manifest = {
        "schema": 1,
        "packages": [],
        "exclusions": [{"name": "git-package", "reason": "git-source"}],
    }

    errors = verify_installed(
        manifest,
        ["git-package"],
        [],
        required_fallback_used=False,
    )

    assert errors == []


def test_ci_wheelhouse_verify_installed_requires_manifest_version(monkeypatch):
    """A required distribution absent from a current manifest is an error."""

    def unexpected_lookup(name):
        raise AssertionError(f"Unexpected installed-version lookup: {name}")

    monkeypatch.setattr("tools.ci_wheelhouse.verify_installed.importlib.metadata.version", unexpected_lookup)

    errors = verify_installed(
        {"schema": 1, "packages": []},
        ["missing-package"],
        [],
        required_fallback_used=False,
    )

    assert errors == ["CI wheelhouse manifest has no version for requested distribution missing-package"]


def test_ci_wheelhouse_verify_installed_allows_required_online_fallback(monkeypatch):
    """A required distribution missing from the manifest is allowed after online fallback."""

    def unexpected_lookup(name):
        raise AssertionError(f"Unexpected installed-version lookup: {name}")

    monkeypatch.setattr("tools.ci_wheelhouse.verify_installed.importlib.metadata.version", unexpected_lookup)

    errors = verify_installed(
        {"schema": 1, "packages": []},
        ["missing-package"],
        [],
        required_fallback_used=True,
    )

    assert errors == []
