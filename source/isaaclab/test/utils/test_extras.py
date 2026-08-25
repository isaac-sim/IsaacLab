# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the missing-extra install hint."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

from isaaclab.utils.extras import EXTRA_FOR_MODULE, missing_extra_hint

_PYPROJECT = Path(__file__).resolve().parents[4] / "pyproject.toml"


def _extras() -> dict[str, set[str]]:
    """Return each optional extra mapped to the distribution names it installs."""
    data = tomllib.loads(_PYPROJECT.read_text())
    extras = data["project"]["optional-dependencies"]
    return {name: {re.split(r"[<>=!;\[ @]", req)[0].strip().lower() for req in reqs} for name, reqs in extras.items()}


def test_every_mapped_extra_exists() -> None:
    """A mapping must not name an extra that pyproject.toml does not declare."""
    declared = set(_extras())
    unknown = {extra for extra in EXTRA_FOR_MODULE.values() if extra not in declared}
    assert not unknown, f"EXTRA_FOR_MODULE names undeclared extras: {sorted(unknown)}"


def test_every_mapped_module_is_installed_by_its_extra() -> None:
    """The extra must actually ship the module it is claimed to provide.

    Distribution names normalize to import names by replacing ``-`` with ``_``. Entries
    whose distribution differs from the import name are listed explicitly so a wrong
    mapping cannot pass silently.
    """
    known_aliases = {"rerun": "rerun-sdk"}
    extras = _extras()
    for module, extra in EXTRA_FOR_MODULE.items():
        dists = extras[extra]
        candidates = {module.replace("_", "-"), module, known_aliases.get(module, module)}
        assert candidates & dists, (
            f"extra '{extra}' does not install anything providing '{module}' (has {sorted(dists)})"
        )


@pytest.mark.parametrize("module", ["rsl_rl", "rlinf"])
def test_deliberate_omissions_stay_unmapped(module: str) -> None:
    """rsl_rl is a base dependency; the rlinf extra ships deps, not the framework."""
    assert module not in EXTRA_FOR_MODULE
    assert missing_extra_hint(module) is None


def test_unknown_module_yields_no_hint() -> None:
    """An unrelated broken import must not be mislabelled as a missing extra."""
    assert missing_extra_hint("numpy") is None


def test_hint_names_module_extra_and_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under uv the hint repeats the invocation with the extra added."""
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    hint = missing_extra_hint("rl_games", command="isaaclab train --rl_library rl_games")
    assert "rl_games is not installed" in hint
    assert "uv run --extra rl-games isaaclab train --rl_library rl_games" in hint


def test_hint_falls_back_to_pip_outside_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without uv markers the hint must not tell a pip/conda user to run uv."""
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    monkeypatch.delenv("UV", raising=False)
    hint = missing_extra_hint("skrl", command="isaaclab train --rl_library skrl")
    assert 'pip install "isaaclab-dev[skrl]"' in hint
    assert "uv run" not in hint
