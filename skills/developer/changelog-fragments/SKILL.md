---
name: isaaclab-writing-changelog-fragments
description: Writes and validates Isaac Lab package changelog fragments using the repository fragment format and bump rules. Use when source package changes need release notes, migration guidance, or changelog validation.
audience: developer
status: stable
owners:
  - isaaclab-maintainers
---

# Writing Changelog Fragments

## When To Use

Use this skill when a PR changes code under `source/<package>/` and needs a changelog fragment, or when reviewing fragment formatting.

Do not use this skill for pure docs, CI, tools, or skills changes unless they also modify `source/<package>/`.

## Workflow

1. Identify each changed package under `source/`.
2. Add fragments for each touched package under `source/<package>/changelog.d/`.
3. Write one `<slug>.<type>.rst` per entry type, where `<type>` is `added`, `changed`,
   `deprecated`, `removed`, or `fixed`; the file holds only that type's `* ` bullets.
4. Add an empty `<slug>.minor` or `<slug>.major` for a minor or major bump (default: patch), or
   only an empty `<slug>.skip` for package changes with no user-facing entry.
5. Include migration guidance for `Deprecated`, `Changed`, and `Removed` entries.
6. Prefix breaking changes with `**Breaking:**`.

## Validation

Run the changelog gate:

```bash
python3 tools/changelog/cli.py check develop
```

Then run the normal formatting gate:

```bash
uv run isaaclab -f
```

## Maintenance

Keep this skill synchronized with `AGENTS.md`, `docs/source/refs/contributing.rst`, and `tools/changelog/`. If changelog policy changes, update those authoritative sources first and keep this skill focused on routing agents to the right workflow.

## References

- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [Changelog tool](../../../tools/changelog/cli.py)
- [Towncrier config](../../../tools/changelog/towncrier.toml)
- [Examples](examples.md)
