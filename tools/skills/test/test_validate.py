# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Skill validation rules."""

from __future__ import annotations

import os
from pathlib import Path

import cli


def _write_skill(root: Path, audience: str = "user", name: str = "isaaclab-testing-skill") -> Path:
    skill_dir = root / audience / "example"
    skill_dir.mkdir(parents=True)
    (skill_dir / "reference.md").write_text("# Reference\n\n## Contents\n\n- Current workflow\n", encoding="utf-8")
    if audience == "user":
        (skill_dir / "evaluations.md").write_text(
            "# Evaluations\n\n"
            "## Contents\n\n- Scenario 1\n- Scenario 2\n- Scenario 3\n\n"
            '## Scenario 1: first\n\nQuery: "Do the first task."\n\nExpected behavior:\n\n- Works.\n\n'
            "Known failure modes:\n\n- Fails.\n\n"
            '## Scenario 2: second\n\nQuery: "Do the second task."\n\nExpected behavior:\n\n- Works.\n\n'
            "Known failure modes:\n\n- Fails.\n\n"
            '## Scenario 3: third\n\nQuery: "Do the third task."\n\nExpected behavior:\n\n- Works.\n\n'
            "Known failure modes:\n\n- Fails.\n",
            encoding="utf-8",
        )
    references = "- [Reference](reference.md)\n"
    if audience == "user":
        references += "- [Evaluations](evaluations.md)\n"
    skill = skill_dir / "SKILL.md"
    skill.write_text(
        "---\n"
        f"name: {name}\n"
        "description: Tests Isaac Lab skill validation behavior. "
        "Use when validating skill fixtures or testing skill rules.\n"
        f"audience: {audience}\n"
        "status: stable\n"
        "owners:\n"
        "  - isaaclab-maintainers\n"
        "---\n\n"
        "# Example Skill\n\n"
        "## When To Use\n\nUse for tests.\n\n"
        "## Workflow\n\n1. Follow the test workflow.\n\n"
        "## Validation\n\nRun the validator.\n\n"
        "## Maintenance\n\nKeep this synchronized with test fixtures.\n\n"
        "## References\n\n"
        f"{references}",
        encoding="utf-8",
    )
    return skill


def _relative_posix_path(path: Path, start: Path) -> Path:
    return Path(os.path.relpath(path, start).replace(os.sep, "/"))


def _write_native_link(path: Path, target: Path, target_is_directory: bool = True) -> None:
    try:
        path.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError):
        path.write_text(target.as_posix(), encoding="utf-8")


def _write_native_aliases(root: Path, skills: list[Path]) -> tuple[Path, Path]:
    agent_root = root / ".agents" / "skills"
    agent_root.mkdir(parents=True)
    for skill in skills:
        metadata, _, error = cli._parse_frontmatter(skill.read_text(encoding="utf-8"))
        assert error is None
        name = metadata["name"]
        assert isinstance(name, str)
        _write_native_link(agent_root / name, _relative_posix_path(skill.parent, agent_root))

    claude_root = root / ".claude" / "skills"
    claude_root.parent.mkdir(parents=True)
    _write_native_link(claude_root, Path("../.agents/skills"))
    return agent_root, claude_root


def test_validate_current_repo_skills():
    assert cli.validate_all() == []


def test_validate_accepts_well_formed_user_skill(tmp_path):
    skill = _write_skill(tmp_path)
    assert cli.Skill(skill).validate() == []


def test_validate_rejects_empty_skill_root(tmp_path):
    errors = cli.validate_all(tmp_path)
    assert any("no skills found" in error for error in errors)


def test_validate_accepts_native_discovery_aliases(tmp_path):
    skill = _write_skill(tmp_path)
    native_roots = _write_native_aliases(tmp_path, [skill])
    assert cli.validate_native_discovery([cli.Skill(skill)], native_roots) == []


def test_validate_accepts_materialized_git_symlink_aliases(tmp_path):
    skill = _write_skill(tmp_path)
    agent_root = tmp_path / ".agents" / "skills"
    agent_root.mkdir(parents=True)
    (agent_root / "isaaclab-testing-skill").write_text(
        _relative_posix_path(skill.parent, agent_root).as_posix(), encoding="utf-8"
    )
    claude_root = tmp_path / ".claude" / "skills"
    claude_root.parent.mkdir(parents=True)
    claude_root.write_text("../.agents/skills", encoding="utf-8")

    assert cli.validate_native_discovery([cli.Skill(skill)], (agent_root, claude_root)) == []


def test_validate_rejects_missing_native_discovery_alias(tmp_path):
    skill = _write_skill(tmp_path)
    native_roots = _write_native_aliases(tmp_path, [skill])
    (native_roots[0] / "isaaclab-testing-skill").unlink()
    errors = cli.validate_native_discovery([cli.Skill(skill)], native_roots)
    assert any("missing native skill alias 'isaaclab-testing-skill'" in error for error in errors)


def test_validate_rejects_native_discovery_alias_with_wrong_target(tmp_path):
    skill = _write_skill(tmp_path)
    native_roots = _write_native_aliases(tmp_path, [skill])
    alias = native_roots[0] / "isaaclab-testing-skill"
    alias.unlink()
    _write_native_link(alias, _relative_posix_path(tmp_path, alias.parent))
    errors = cli.validate_native_discovery([cli.Skill(skill)], native_roots)
    assert any("native skill alias resolves to" in error for error in errors)


def test_validate_rejects_duplicate_names(tmp_path):
    _write_skill(tmp_path, audience="user", name="isaaclab-duplicate-skill")
    _write_skill(tmp_path, audience="developer", name="isaaclab-duplicate-skill")
    errors = cli.validate_all(tmp_path)
    assert any("duplicate skill name" in error for error in errors)


def test_validate_rejects_unknown_cross_skill_reference(tmp_path):
    skill = _write_skill(tmp_path, name="isaaclab-source-skill")
    text = skill.read_text(encoding="utf-8").replace(
        "## Workflow\n\n1. Follow the test workflow.\n\n",
        "## Workflow\n\n1. Pair this with `isaaclab-missing-skill`.\n\n",
    )
    skill.write_text(text, encoding="utf-8")
    errors = cli.validate_all(tmp_path)
    assert any("cross-skill reference 'isaaclab-missing-skill'" in error for error in errors)


def test_validate_accepts_known_cross_skill_reference(tmp_path):
    source = _write_skill(tmp_path, audience="user", name="isaaclab-source-skill")
    text = source.read_text(encoding="utf-8").replace(
        "## Workflow\n\n1. Follow the test workflow.\n\n",
        "## Workflow\n\n1. Pair this with `isaaclab-target-skill`.\n\n",
    )
    source.write_text(text, encoding="utf-8")
    target_dir = tmp_path / "developer" / "example"
    target_dir.mkdir(parents=True)
    (target_dir / "reference.md").write_text("# Reference\n\n## Contents\n\n- Workflow\n", encoding="utf-8")
    (target_dir / "SKILL.md").write_text(
        "---\n"
        "name: isaaclab-target-skill\n"
        "description: Tests Isaac Lab cross-skill routing. Use when validating cross-skill references.\n"
        "audience: developer\n"
        "status: stable\n"
        "owners:\n"
        "  - isaaclab-maintainers\n"
        "---\n\n"
        "# Target Skill\n\n"
        "## When To Use\n\nUse for tests.\n\n"
        "## Workflow\n\n1. Follow the test workflow.\n\n"
        "## Validation\n\nRun the validator.\n\n"
        "## Maintenance\n\nKeep this synchronized with test fixtures.\n\n"
        "## References\n\n- [Reference](reference.md)\n",
        encoding="utf-8",
    )
    assert not any("cross-skill reference" in error for error in cli.validate_all(tmp_path))


def test_validate_rejects_missing_required_section(tmp_path):
    skill = _write_skill(tmp_path)
    text = skill.read_text(encoding="utf-8").replace("## Validation\n\nRun the validator.\n\n", "")
    skill.write_text(text, encoding="utf-8")
    errors = cli.Skill(skill).validate()
    assert any("missing required sections" in error for error in errors)


def test_validate_rejects_broken_link(tmp_path):
    skill = _write_skill(tmp_path)
    text = skill.read_text(encoding="utf-8").replace("[Reference](reference.md)", "[Reference](missing.md)")
    skill.write_text(text, encoding="utf-8")
    errors = cli.Skill(skill).validate()
    assert any("broken link" in error for error in errors)


def test_validate_rejects_user_skill_without_three_evaluations(tmp_path):
    skill = _write_skill(tmp_path)
    (skill.parent / "evaluations.md").write_text(
        "# Evaluations\n\n## Scenario 1: only\n\nExpected behavior:\n\n- Works.\n",
        encoding="utf-8",
    )
    errors = cli.Skill(skill).validate()
    assert any("at least three evaluation scenarios" in error for error in errors)


def test_validate_rejects_user_skill_without_evaluations_link(tmp_path):
    skill = _write_skill(tmp_path)
    text = skill.read_text(encoding="utf-8").replace(
        "- [Evaluations](evaluations.md)\n",
        "Mention evaluations.md in prose without linking it.\n",
    )
    skill.write_text(text, encoding="utf-8")
    errors = cli.Skill(skill).validate()
    assert any("must link to evaluations.md" in error for error in errors)


def test_validate_accepts_dot_slash_evaluations_link(tmp_path):
    skill = _write_skill(tmp_path)
    text = skill.read_text(encoding="utf-8").replace(
        "- [Evaluations](evaluations.md)\n",
        "- [Evaluations](./evaluations.md)\n",
    )
    skill.write_text(text, encoding="utf-8")
    errors = cli.Skill(skill).validate()
    assert not any("must link to evaluations.md" in error for error in errors)


def test_validate_rejects_user_skill_without_evaluation_details(tmp_path):
    skill = _write_skill(tmp_path)
    (skill.parent / "evaluations.md").write_text(
        "# Evaluations\n\n"
        "## Scenario 1: first\n\n- Works.\n\n"
        "## Scenario 2: second\n\n- Works.\n\n"
        "## Scenario 3: third\n\n- Works.\n",
        encoding="utf-8",
    )
    errors = cli.Skill(skill).validate()
    assert any("sample query" in error for error in errors)
    assert any("expected behavior" in error for error in errors)
    assert any("known failure modes" in error for error in errors)


def test_validate_rejects_scenario_without_evaluation_details(tmp_path):
    skill = _write_skill(tmp_path)
    text = (
        (skill.parent / "evaluations.md")
        .read_text(encoding="utf-8")
        .replace(
            '## Scenario 2: second\n\nQuery: "Do the second task."\n\nExpected behavior:\n\n- Works.\n\n'
            "Known failure modes:\n\n- Fails.\n\n",
            "## Scenario 2: second\n\n- Missing details.\n\n",
        )
    )
    (skill.parent / "evaluations.md").write_text(text, encoding="utf-8")
    errors = cli.Skill(skill).validate()
    assert any("Scenario 2" in error and "sample query" in error for error in errors)
    assert any("Scenario 2" in error and "expected behavior" in error for error in errors)
    assert any("Scenario 2" in error and "known failure modes" in error for error in errors)


def test_validate_rejects_frontmatter_block_scalars(tmp_path):
    skill = _write_skill(tmp_path)
    text = skill.read_text(encoding="utf-8").replace(
        "description: Tests Isaac Lab skill validation behavior. "
        "Use when validating skill fixtures or testing skill rules.\n",
        "description: |\n"
        "  Tests Isaac Lab skill validation behavior. "
        "Use when validating skill fixtures or testing skill rules.\n",
    )
    skill.write_text(text, encoding="utf-8")
    errors = cli.Skill(skill).validate()
    assert any("unsupported block scalar" in error for error in errors)


def test_validate_rejects_first_person_description(tmp_path):
    skill = _write_skill(tmp_path)
    text = skill.read_text(encoding="utf-8").replace(
        "description: Tests Isaac Lab skill validation behavior. "
        "Use when validating skill fixtures or testing skill rules.",
        "description: I can help validate skills. Use when testing skill rules.",
    )
    skill.write_text(text, encoding="utf-8")
    errors = cli.Skill(skill).validate()
    assert any("third person" in error for error in errors)


def test_validate_rejects_windows_paths(tmp_path):
    skill = _write_skill(tmp_path)
    text = skill.read_text(encoding="utf-8") + "\nUse scripts\\helper.py for this task.\n"
    skill.write_text(text, encoding="utf-8")
    errors = cli.Skill(skill).validate()
    assert any("forward-slash paths" in error for error in errors)
