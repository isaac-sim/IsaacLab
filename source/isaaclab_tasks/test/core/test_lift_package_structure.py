# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for the unified core Lift task package."""

from pathlib import Path

_TASKS_PACKAGE = Path(__file__).parents[2] / "isaaclab_tasks"
_LIFT_PACKAGE = _TASKS_PACKAGE / "core" / "lift"

_REQUIRED_DEXTEROUS_LIFT_FILES = {
    "adr_curriculum.py",
    "lift_env_cfg.py",
    "config/franka/franka_env_cfg.py",
    "config/kuka_allegro/agents/models.py",
    "config/kuka_allegro/camera_cfg.py",
    "config/kuka_allegro/kuka_allegro_camera_env_cfg.py",
    "config/kuka_allegro/kuka_allegro_env_cfg.py",
    "mdp/commands/pose_commands.py",
    "mdp/commands/pose_commands_cfg.py",
    "mdp/curriculums.py",
    "mdp/events.py",
    "mdp/events_cfg.py",
    "mdp/observations.py",
    "mdp/rewards.py",
    "mdp/terminations.py",
    "mdp/utils.py",
}


def test_lift_owns_the_complete_dexterous_task_tree() -> None:
    """The unified package must own every module needed by current dexterous training."""
    actual_files = {path.relative_to(_LIFT_PACKAGE).as_posix() for path in _LIFT_PACKAGE.rglob("*.py")}

    assert actual_files >= _REQUIRED_DEXTEROUS_LIFT_FILES


def test_dexsuite_package_and_dependencies_are_absent() -> None:
    """DexSuite must not survive as a package, alias, or dependency of core Lift."""
    assert not (_TASKS_PACKAGE / "core" / "dexsuite").exists()

    offenders = []
    for path in _LIFT_PACKAGE.rglob("*"):
        if path.suffix not in {".py", ".pyi"}:
            continue
        source = path.read_text(encoding="utf-8")
        if "dexsuite" in path.name.casefold() or "dexsuite" in source.casefold():
            offenders.append(path.relative_to(_TASKS_PACKAGE).as_posix())

    assert offenders == []
