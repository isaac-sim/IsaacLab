# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the dependency-impact classifier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("dependency_impact.py")
_SPEC = importlib.util.spec_from_file_location("dependency_impact", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
# Register before exec so ``@dataclass`` can resolve the module's namespace (matches a
# normal ``import dependency_impact``).
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _make_package(repo_root: Path, name: str, dependencies: list[str], files: list[str]) -> None:
    """Create a fake extension package with an ``extension.toml`` and some ``*.py`` files.

    Args:
        repo_root: Temporary repository root.
        name: Extension directory name (also its dependency identifier).
        dependencies: Names of extensions this package depends on.
        files: Package-relative file paths to create (only ``*.py`` matter to the graph).
    """
    package_dir = repo_root / "source" / name
    (package_dir / "config").mkdir(parents=True, exist_ok=True)
    deps_block = "".join(f'"{dep}" = {{}}\n' for dep in dependencies)
    (package_dir / "config" / "extension.toml").write_text(
        f'[dependencies]\n{deps_block}\n[[python.module]]\nname = "{name}"\n', encoding="utf-8"
    )
    for rel in files:
        target = package_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# stub\n", encoding="utf-8")


def _build_chain(repo_root: Path) -> None:
    """Build a diamond graph: base <- mid_a, mid_b ; mid_a, mid_b <- top."""
    _make_package(repo_root, "base", [], ["core.py"])
    _make_package(repo_root, "mid_a", ["base"], ["a.py"])
    _make_package(repo_root, "mid_b", ["base"], ["b.py"])
    _make_package(repo_root, "top", ["mid_a", "mid_b"], ["t.py"])


def test_changed_bucket_lists_only_changed_python(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    impact = _MODULE.build_impact(
        ["source/base/core.py", "source/base/README.md"],
        repo_root=tmp_path,
    )
    assert impact["changed"]["python"] == ["source/base/core.py"]


def test_direct_dependents_are_separated_from_transitive(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    impact = _MODULE.build_impact(["source/base/core.py"], repo_root=tmp_path)

    assert impact["dependents"]["python"] == ["source/mid_a/a.py", "source/mid_b/b.py"]
    assert impact["transitive"]["python"] == ["source/top/t.py"]


def test_buckets_are_disjoint(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    impact = _MODULE.build_impact(["source/base/core.py"], repo_root=tmp_path)

    changed = set(impact["changed"]["python"])
    dependents = set(impact["dependents"]["python"])
    transitive = set(impact["transitive"]["python"])
    assert changed.isdisjoint(dependents)
    assert changed.isdisjoint(transitive)
    assert dependents.isdisjoint(transitive)


def test_package_that_is_both_direct_and_indirect_counts_as_direct(tmp_path: Path) -> None:
    # ``top`` depends on ``base`` directly AND through ``mid``; it must be a direct dependent.
    _make_package(tmp_path, "base", [], ["core.py"])
    _make_package(tmp_path, "mid", ["base"], ["m.py"])
    _make_package(tmp_path, "top", ["base", "mid"], ["t.py"])

    impact = _MODULE.build_impact(["source/base/core.py"], repo_root=tmp_path)
    assert impact["dependents"]["python"] == ["source/mid/m.py", "source/top/t.py"]
    assert impact["transitive"]["python"] == []


def test_change_to_leaf_package_has_no_dependents(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    impact = _MODULE.build_impact(["source/top/t.py"], repo_root=tmp_path)
    assert impact["dependents"]["python"] == []
    assert impact["transitive"]["python"] == []


def test_untracked_change_yields_empty_buckets(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    impact = _MODULE.build_impact(["docs/index.rst", "tools/foo.py"], repo_root=tmp_path)
    assert impact["changed"]["python"] == ["tools/foo.py"]
    assert impact["dependents"]["python"] == []
    assert impact["transitive"]["python"] == []


def test_non_python_change_still_seeds_dependents(tmp_path: Path) -> None:
    # A change to a non-Python file inside a package (here ``base``'s extension.toml) must still
    # seed the reverse-dependency walk, even though no ``.py`` file was touched.
    _build_chain(tmp_path)
    impact = _MODULE.build_impact(["source/base/config/extension.toml"], repo_root=tmp_path)
    assert impact["changed"]["python"] == []
    assert impact["dependents"]["python"] == ["source/mid_a/a.py", "source/mid_b/b.py"]
    assert impact["transitive"]["python"] == ["source/top/t.py"]


def test_windows_paths_are_normalized(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    impact = _MODULE.build_impact([r"source\base\core.py"], repo_root=tmp_path)
    assert impact["changed"]["python"] == ["source/base/core.py"]
    assert impact["dependents"]["python"] == ["source/mid_a/a.py", "source/mid_b/b.py"]
