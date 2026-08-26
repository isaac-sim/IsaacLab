# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate Isaac Lab agent skills.

Skills live under ``skills/<audience>/<slug>/SKILL.md`` and are markdown
guidance assets for agents. This tool keeps the accepted skill format small,
portable, and reviewable.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).parent.parent.parent
SKILLS_ROOT = REPO_ROOT / "skills"
AGENT_SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
CLAUDE_SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"
NATIVE_SKILLS_ROOTS = (AGENT_SKILLS_ROOT, CLAUDE_SKILLS_ROOT)

INTERNAL_DIR = "_internal"
AUDIENCES = {"user"}
STATUSES = {"experimental", "stable", "deprecated"}
REQUIRED_SECTIONS = {"When To Use", "Workflow", "Validation", "Maintenance", "References"}

NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
HEADING_RE = re.compile(r"^## (?P<title>.+?)\s*$", re.MULTILINE)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target>[^)]+)\)")
BACKTICK_PATH_RE = re.compile(r"`(?P<path>(?:docs|source|scripts|skills|tools|\.github)/[^`\s]+)`")
CROSS_SKILL_RE = re.compile(r"`(?P<name>isaaclab-[a-z0-9-]+)`")
XML_TAG_RE = re.compile(r"<[^>]+>")
WINDOWS_PATH_RE = re.compile(r"(?<!`)[A-Za-z0-9_.-]+\\[A-Za-z0-9_.-]+")
SCENARIO_RE = re.compile(r"^#{2,3} (?P<title>Scenario\b.*)$", re.MULTILINE)
GENERIC_NAME_PARTS = {"helper", "helpers", "utils", "tools"}


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _native_symlink_target(path: Path) -> Path | None:
    """Return the resolved target for native discovery symlinks.

    Git symlinks may be checked out as one-line target files on Windows when
    symlink creation is unavailable. Accept that materialized form so the
    validator can still check the recorded target in local Windows checkouts.
    """
    if path.is_symlink():
        return path.resolve()
    if not path.is_file():
        return None

    target_text = path.read_text(encoding="utf-8")
    if not target_text or "\n" in target_text or "\r" in target_text:
        return None
    return (path.parent / target_text).resolve()


def _parse_frontmatter(text: str) -> tuple[dict[str, str | list[str]], str, str | None]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}, text, "missing YAML frontmatter"

    end = None
    for index, line in enumerate(lines[1:], start=1):
        if line == "---":
            end = index
            break
    if end is None:
        return {}, text, "unterminated YAML frontmatter"

    data: dict[str, str | list[str]] = {}
    current_list_key: str | None = None
    current_map_key: str | None = None
    for line in lines[1:end]:
        if not line.strip():
            continue
        if line.startswith("  - "):
            if current_list_key is None:
                return data, "\n".join(lines[end + 1 :]), f"list item without key: {line!r}"
            value = line[4:].strip()
            current = data.setdefault(current_list_key, [])
            if not isinstance(current, list):
                return data, "\n".join(lines[end + 1 :]), f"mixed scalar/list value for {current_list_key!r}"
            current.append(value)
            continue
        if line.startswith("  ") and current_map_key is not None and ":" in line:
            # Nested `parent:\n  key: value` map entry. Flattened as `parent.key` so callers can
            # distinguish it from a mistaken top-level `key: value`.
            nested_key, nested_value = line.strip().split(":", 1)
            data[f"{current_map_key}.{nested_key.strip()}"] = nested_value.strip().strip("\"'")
            continue
        current_list_key = None
        current_map_key = None
        if ":" not in line:
            return data, "\n".join(lines[end + 1 :]), f"invalid frontmatter line: {line!r}"
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return data, "\n".join(lines[end + 1 :]), "empty frontmatter key"
        if value in {"|", ">"}:
            return data, "\n".join(lines[end + 1 :]), f"unsupported block scalar for {key!r}"
        if value:
            data[key] = value.strip("\"'")
        else:
            data[key] = []
            current_list_key = key
            current_map_key = key

    return data, "\n".join(lines[end + 1 :]), None


@dataclass(frozen=True)
class Skill:
    """One ``SKILL.md`` file and its validation rules."""

    path: Path

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def audience_dir(self) -> str:
        try:
            return self.path.relative_to(SKILLS_ROOT).parts[0]
        except ValueError:
            return self.path.parent.parent.name

    @property
    def is_internal(self) -> bool:
        return self.audience_dir == INTERNAL_DIR

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.path.name != "SKILL.md":
            return [f"{_display_path(self.path)}: expected file named SKILL.md"]
        if not self.path.exists():
            return [f"{_display_path(self.path)}: file does not exist"]

        text = self.path.read_text(encoding="utf-8")
        metadata, body, frontmatter_error = _parse_frontmatter(text)
        if frontmatter_error:
            errors.append(f"{_display_path(self.path)}: {frontmatter_error}")
            return errors

        errors.extend(self._validate_metadata(metadata))
        errors.extend(self._validate_body(body))
        errors.extend(self._validate_links(self.path, body, enforce_one_level=True))
        errors.extend(self._validate_backtick_paths(self.path, body))
        errors.extend(self._validate_reference_files())
        if not self.is_internal:
            errors.extend(self._validate_user_evaluations(body))
        errors.extend(self._validate_scripts(body))
        return errors

    def _validate_metadata(self, metadata: dict[str, str | list[str]]) -> list[str]:
        errors: list[str] = []
        name = metadata.get("name")
        description = metadata.get("description")
        license_ = metadata.get("license")
        # The parser flattens `metadata:\n  author:` into a dotted `metadata.author` key so a
        # mistaken top-level `author:` (not nested under `metadata:`) can be told apart from it.
        author = metadata.get("metadata.author")
        stray_top_level_author = metadata.get("author")

        if not isinstance(name, str) or not name:
            errors.append(f"{_display_path(self.path)}: missing required frontmatter field 'name'")
        elif not NAME_RE.fullmatch(name):
            errors.append(f"{_display_path(self.path)}: invalid skill name {name!r}")
        else:
            name_parts = set(name.split("-"))
            if name_parts & GENERIC_NAME_PARTS:
                errors.append(
                    f"{_display_path(self.path)}: skill name contains generic term {name_parts & GENERIC_NAME_PARTS}"
                )
            if "claude" in name_parts or "anthropic" in name_parts:
                errors.append(f"{_display_path(self.path)}: skill name uses reserved words")

        if not isinstance(description, str) or not description:
            errors.append(f"{_display_path(self.path)}: missing required frontmatter field 'description'")
        elif len(description) > 1024:
            errors.append(f"{_display_path(self.path)}: description is longer than 1024 characters")
        else:
            lowered = description.lower()
            if XML_TAG_RE.search(description):
                errors.append(f"{_display_path(self.path)}: description must not contain XML tags")
            if "use when" not in lowered:
                errors.append(f"{_display_path(self.path)}: description must include 'Use when' trigger guidance")
            if lowered.startswith(("i ", "i can", "you ", "you can")):
                errors.append(f"{_display_path(self.path)}: description must be written in third person")

        if not isinstance(license_, str) or not license_:
            errors.append(f"{_display_path(self.path)}: missing required frontmatter field 'license'")

        if not isinstance(author, str) or not author:
            if isinstance(stray_top_level_author, str) and stray_top_level_author:
                errors.append(f"{_display_path(self.path)}: 'author' must be nested under 'metadata:', not top-level")
            else:
                errors.append(f"{_display_path(self.path)}: missing required frontmatter field 'metadata.author'")

        if self.is_internal:
            return errors

        # Fields required only for published (non-internal) skills.
        audience = metadata.get("audience")
        status = metadata.get("status")
        owners = metadata.get("owners")

        if audience not in AUDIENCES:
            errors.append(f"{_display_path(self.path)}: audience must be one of {sorted(AUDIENCES)}")
        elif audience != self.audience_dir:
            errors.append(
                f"{_display_path(self.path)}: audience {audience!r} does not match directory {self.audience_dir!r}"
            )

        if status not in STATUSES:
            errors.append(f"{_display_path(self.path)}: status must be one of {sorted(STATUSES)}")

        if not isinstance(owners, list) or not owners or any(not owner for owner in owners):
            errors.append(f"{_display_path(self.path)}: owners must list at least one owner")

        return errors

    def _validate_body(self, body: str) -> list[str]:
        errors: list[str] = []
        line_count = len(body.splitlines())
        if line_count > 500:
            errors.append(f"{_display_path(self.path)}: SKILL.md body must stay under 500 lines")
        if WINDOWS_PATH_RE.search(body):
            errors.append(f"{_display_path(self.path)}: use forward-slash paths, not Windows-style paths")

        sections = {match.group("title") for match in HEADING_RE.finditer(body)}
        missing = sorted(REQUIRED_SECTIONS - sections)
        if missing:
            errors.append(f"{_display_path(self.path)}: missing required sections: {', '.join(missing)}")
        return errors

    def _validate_links(self, source: Path, body: str, enforce_one_level: bool = False) -> list[str]:
        errors: list[str] = []
        for match in LINK_RE.finditer(body):
            raw_target = match.group("target").strip()
            if self._is_external_or_anchor(raw_target):
                continue
            target_text = unquote(raw_target.split("#", 1)[0])
            target = (source.parent / target_text).resolve()
            if not target.exists():
                errors.append(f"{_display_path(source)}: broken link {raw_target!r}")
                continue
            if enforce_one_level and _is_relative_to(target, self.root):
                rel = target.relative_to(self.root)
                if len(rel.parts) > 1:
                    errors.append(
                        f"{_display_path(source)}: local skill references must be one level deep: {raw_target}"
                    )
        return errors

    def _validate_reference_files(self) -> list[str]:
        errors: list[str] = []
        for reference in self.root.rglob("*.md"):
            if reference.name == "SKILL.md":
                continue
            text = reference.read_text(encoding="utf-8")
            lines = text.splitlines()
            if len(lines) > 100 and not any(line.strip() in {"# Contents", "## Contents"} for line in lines[:20]):
                errors.append(f"{_display_path(reference)}: reference files over 100 lines need a Contents section")
            if WINDOWS_PATH_RE.search(text):
                errors.append(f"{_display_path(reference)}: use forward-slash paths, not Windows-style paths")
            errors.extend(self._validate_links(reference, text))
            errors.extend(self._validate_backtick_paths(reference, text))
        return errors

    def _validate_backtick_paths(self, source: Path, text: str) -> list[str]:
        errors: list[str] = []
        for match in BACKTICK_PATH_RE.finditer(text):
            raw_path = match.group("path").rstrip(".,);:")
            if "<" in raw_path or ">" in raw_path:
                continue
            target = (REPO_ROOT / raw_path).resolve()
            if not target.exists():
                errors.append(f"{_display_path(source)}: backtick path does not exist: {raw_path}")
        return errors

    def _validate_user_evaluations(self, body: str) -> list[str]:
        errors: list[str] = []
        evaluations = self.root / "evaluations.md"
        evals_json = self.root / "evals" / "evals.json"

        if not evaluations.exists() and not evals_json.exists():
            errors.append(
                f"{_display_path(self.path)}: user-facing skills must include evaluations.md or evals/evals.json"
            )
            return errors

        # Validate evaluations.md when present.
        if evaluations.exists():
            has_evaluations_link = any(
                unquote(match.group("target").strip().split("#", 1)[0]).removeprefix("./") == "evaluations.md"
                for match in LINK_RE.finditer(body)
            )
            if not has_evaluations_link:
                errors.append(
                    f"{_display_path(self.path)}: user-facing skills must link to evaluations.md from SKILL.md"
                )
            text = evaluations.read_text(encoding="utf-8")
            scenario_matches = list(SCENARIO_RE.finditer(text))
            if len(scenario_matches) < 3:
                errors.append(
                    f"{_display_path(evaluations)}: user-facing skills need at least three evaluation scenarios"
                )
            for index, match in enumerate(scenario_matches, start=1):
                next_match = scenario_matches[index] if index < len(scenario_matches) else None
                scenario_text = text[match.start() : next_match.start() if next_match else len(text)]
                scenario_name = match.group("title").strip()
                if "Query:" not in scenario_text:
                    errors.append(f"{_display_path(evaluations)}: {scenario_name} must include a sample query")
                if "Expected behavior:" not in scenario_text:
                    errors.append(f"{_display_path(evaluations)}: {scenario_name} must include expected behavior")
                if (
                    "Known failure modes:" not in scenario_text
                    and "Pass/fail" not in scenario_text
                    and "pass/fail" not in scenario_text
                ):
                    errors.append(
                        f"{_display_path(evaluations)}: {scenario_name} must include known failure modes "
                        "or pass/fail criteria"
                    )

        # Validate evals/evals.json when present.
        if evals_json.exists():
            import json

            try:
                data = json.loads(evals_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"{_display_path(evals_json)}: invalid JSON: {exc}")
                return errors
            evals = data.get("evals", []) if isinstance(data, dict) else []
            valid_entries = [entry for entry in evals if isinstance(entry, dict)]
            if len(valid_entries) < 2:
                errors.append(f"{_display_path(evals_json)}: must contain at least two eval entries")
            for entry in evals:
                if not isinstance(entry, dict):
                    errors.append(f"{_display_path(evals_json)}: eval entry must be an object, got {entry!r}")
                    continue
                for field in ("id", "prompt"):
                    if not entry.get(field):
                        errors.append(f"{_display_path(evals_json)}: entry missing required field '{field}'")
                # `expected_skill: null` is intentional for negative eval cases (no skill should
                # trigger); only its key membership is required, not a truthy value.
                if "expected_skill" not in entry:
                    errors.append(f"{_display_path(evals_json)}: entry missing required field 'expected_skill'")
                elif entry["expected_skill"] == "":
                    errors.append(
                        f"{_display_path(evals_json)}: 'expected_skill' must not be an empty string"
                        " (use null for negative cases)"
                    )
                if "assertions" not in entry or not isinstance(entry["assertions"], list):
                    errors.append(
                        f"{_display_path(evals_json)}: entry {entry.get('id', '?')!r} missing 'assertions' list"
                    )

        return errors

    def _validate_scripts(self, body: str) -> list[str]:
        scripts = self.root / "scripts"
        if not scripts.exists():
            return []
        if "Run `scripts/" not in body and "Read `scripts/" not in body:
            return [f"{_display_path(self.path)}: scripts require explicit run-or-read instructions"]
        return []

    @staticmethod
    def _is_external_or_anchor(target: str) -> bool:
        return target.startswith(("#", "http://", "https://", "mailto:"))


def iter_skills(root: Path = SKILLS_ROOT) -> list[Skill]:
    if not root.exists():
        return []
    return [Skill(path) for path in sorted(root.glob("*/*/SKILL.md"))]


def iter_public_skills(root: Path = SKILLS_ROOT) -> list[Skill]:
    """Return only published (non-internal) skills."""
    return [s for s in iter_skills(root) if not s.is_internal]


def validate_native_discovery(skills: list[Skill], native_roots: tuple[Path, ...] = NATIVE_SKILLS_ROOTS) -> list[str]:
    """Validate native agent aliases for the canonical repository skills."""
    errors: list[str] = []
    expected: dict[str, Path] = {}
    for skill in skills:
        metadata, _, frontmatter_error = _parse_frontmatter(skill.path.read_text(encoding="utf-8"))
        name = metadata.get("name")
        if frontmatter_error is None and isinstance(name, str):
            expected[name] = skill.root.resolve()

    for native_root in native_roots:
        display_root = _display_path(native_root)
        root_target = _native_symlink_target(native_root)
        discovery_root = root_target or native_root
        if not discovery_root.exists():
            errors.append(f"{display_root}: native skill discovery directory does not exist")
            continue
        if not discovery_root.is_dir():
            errors.append(f"{display_root}: native skill discovery path must be a directory")
            continue

        aliases = {alias.name: alias for alias in discovery_root.iterdir()}
        missing = sorted(set(expected) - set(aliases))
        unexpected = sorted(set(aliases) - set(expected))
        for name in missing:
            errors.append(f"{display_root}: missing native skill alias {name!r}")
        for name in unexpected:
            errors.append(f"{display_root}: unexpected native skill alias {name!r}")

        for name, skill_root in expected.items():
            alias = aliases.get(name)
            if alias is None:
                continue
            alias_target = _native_symlink_target(alias)
            if alias_target is None:
                errors.append(f"{_display_path(alias)}: native skill alias must be a symbolic link")
                continue
            if alias_target != skill_root:
                errors.append(
                    f"{_display_path(alias)}: native skill alias resolves to {_display_path(alias_target)}, "
                    f"expected {_display_path(skill_root)}"
                )

    claude_target = _native_symlink_target(CLAUDE_SKILLS_ROOT)
    if (
        native_roots == NATIVE_SKILLS_ROOTS
        and CLAUDE_SKILLS_ROOT.exists()
        and (claude_target is None or claude_target != AGENT_SKILLS_ROOT.resolve())
    ):
        errors.append(
            f"{_display_path(CLAUDE_SKILLS_ROOT)}: Claude discovery must link to {_display_path(AGENT_SKILLS_ROOT)}"
        )
    return errors


def validate_all(root: Path = SKILLS_ROOT) -> list[str]:
    errors: list[str] = []
    skills = iter_skills(root)
    if not skills:
        errors.append(f"{_display_path(root)}: no skills found; expected */*/SKILL.md")
    names: dict[str, Path] = {}
    bodies: list[tuple[Skill, str]] = []
    for skill in skills:
        metadata, body, frontmatter_error = _parse_frontmatter(skill.path.read_text(encoding="utf-8"))
        if frontmatter_error is None and isinstance(metadata.get("name"), str):
            name = str(metadata["name"])
            if name in names:
                errors.append(
                    f"{_display_path(skill.path)}: duplicate skill name {name!r}; "
                    f"first used by {_display_path(names[name])}"
                )
            else:
                names[name] = skill.path
        bodies.append((skill, body))
        errors.extend(skill.validate())

    for skill, body in bodies:
        for match in CROSS_SKILL_RE.finditer(body):
            referenced = match.group("name")
            if referenced not in names:
                errors.append(
                    f"{_display_path(skill.path)}: cross-skill reference {referenced!r} "
                    "does not match any registered skill name"
                )
    if root.resolve() == SKILLS_ROOT.resolve():
        errors.extend(validate_native_discovery(skills))
    return errors


def _cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else SKILLS_ROOT
    errors = validate_all(root)
    if errors:
        print("Skill validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Validated {len(iter_skills(root))} skills.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="validate Isaac Lab skills")
    check.add_argument("--root", type=Path, default=None, help="skills root to validate")
    check.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
