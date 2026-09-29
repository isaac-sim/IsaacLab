---
name: isaaclab-following-coding-style
description: Applies Isaac Lab coding style, API design, docstring, type-hint, lazy export, and contribution conventions. Use when writing or reviewing Isaac Lab Python code, public APIs, config classes, module exports, or documentation strings.
audience: developer
status: stable
owners:
  - isaaclab-maintainers
---

# Following Coding Style

## When To Use

Use this skill when adding or reviewing Isaac Lab code, especially public APIs, config classes, package exports, docstrings, type hints, or files that may import simulator-dependent modules.

Do not use this skill as a replacement for the contribution guide. Read the authoritative docs before making broad style decisions.

## Workflow

1. Read the `Coding Style` section of `docs/source/refs/contributing.rst` and apply the relevant subsections.
2. Check `AGENTS.md` and any more-specific instructions for repository workflow constraints.
3. Inspect surrounding code and the guide's ordering rules before choosing local structure; its minimal
   function example illustrates signatures and docstrings.
4. For validation, follow the guide's `Unit Testing` and `Tools` sections. For test changes, apply
   [the test-audit skill](../test-audit/SKILL.md) before adding or removing coverage.

## Validation

Run formatting and lint checks:

```bash
uv run isaaclab -f
```

For focused tests, use:

```bash
uv run python -m pytest PATH_TO_TEST
```

For skill changes, run:

```bash
uv run --no-project python tools/skills/cli.py check
```

## Maintenance

Keep this skill synchronized with `AGENTS.md`, `docs/source/refs/contributing.rst`, `docs/source/refs/snippets/code_skeleton.py`, and `.pre-commit-config.yaml`. If coding-style guidance changes, update those authoritative files first and keep this skill as a routing checklist.

## References

- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [Minimal function example](../../../docs/source/refs/snippets/code_skeleton.py)
- [Examples](examples.md)
