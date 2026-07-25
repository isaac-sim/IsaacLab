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

1. Read the `Coding Style` section of `docs/source/refs/contributing.rst`.
2. Check `AGENTS.md` for always-on Isaac Lab API, docstring, dependency, and tooling rules.
3. Keep the change aligned with local patterns in the touched package.
4. Use the documented file and class member ordering.
5. Avoid runtime circular imports with `TYPE_CHECKING`, resolvable strings, and lazy exports where the contribution guide calls for them.
6. Use modern, specific type hints and Google-style docstrings.
7. Keep public-facing docs and auto-generated API docs current when adding public symbols.
8. Run the formatting and lint feedback loop.

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
- [Code skeleton](../../../docs/source/refs/snippets/code_skeleton.py)
- [Examples](examples.md)
