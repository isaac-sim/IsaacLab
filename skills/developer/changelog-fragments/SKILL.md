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
2. Add one fragment per touched package under `source/<package>/changelog.d/`.
3. Choose the suffix:
   - `<slug>.rst` for patch changes.
   - `<slug>.minor.rst` for minor changes.
   - `<slug>.major.rst` for major changes.
   - `<slug>.skip` for package changes that intentionally have no user-facing entry.
4. Place entries under `Added`, `Changed`, `Deprecated`, `Removed`, or `Fixed`.
5. Include migration guidance for `Deprecated`, `Changed`, and `Removed` entries.
6. Prefix breaking changes with `**Breaking:**`.

## Validation

Run the changelog gate:

```bash
./isaaclab.sh -p tools/changelog/cli.py check develop
```

Then run the normal formatting gate:

```bash
./isaaclab.sh -f
```

## Maintenance

Keep this skill synchronized with `AGENTS.md`, `docs/source/refs/contributing.rst`, and `tools/changelog/cli.py`. If changelog policy changes, update those authoritative sources first and keep this skill focused on routing agents to the right workflow.

## References

- [Contributing guide](../../../docs/source/refs/contributing.rst)
- [Changelog tool](../../../tools/changelog/cli.py)
- [Integration examples](../../../tools/changelog/test/integration/README.md)
- [Examples](examples.md)
