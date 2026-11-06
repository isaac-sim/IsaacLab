---
name: isaaclab-following-coding-style
description: Applies Isaac Lab Python conventions. Use when writing or reviewing APIs, config classes, exports, and docstrings.
audience: developer
status: stable
owners:
  - isaaclab-maintainers
---

# Following Coding Style

## When To Use

Use for Isaac Lab Python changes and reviews, especially APIs and imports that must remain usable before simulator startup.

## Workflow

1. Apply the relevant [Coding Style subsections](../../../docs/source/refs/contributing.rst#coding-style), reusing guidance already loaded and still current.
2. Match the surrounding API and import lifecycle. Use the guide's minimal function example when signature or docstring conventions are unclear.
3. For test changes, apply [the authoring gate](../test-audit/SKILL.md).

## Validation

Follow the guide's [Unit Testing](../../../docs/source/refs/contributing.rst#unit-testing) and [Tools](../../../docs/source/refs/contributing.rst#tools) sections for focused checks and final validation. Skill-only changes use the [skill validator](../../../tools/skills/cli.py):

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `AGENTS.md`, `docs/source/refs/contributing.rst`, `docs/source/refs/snippets/code_skeleton.py`, and `.pre-commit-config.yaml`. If coding-style guidance changes, update those authoritative files first and keep this skill as a routing checklist.

## References

- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [Minimal function example](../../../docs/source/refs/snippets/code_skeleton.py)
- [Examples](examples.md)
