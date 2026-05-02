# Changelog integration fixtures

End-to-end test fixtures for `tools/changelog/cli.py compile`. Each
subdirectory holds a worked example: input fragments, the starting
`CHANGELOG.rst`, and the expected compiled output.

`tools/changelog/test/test_integration.py` runs the compiler
against each one and asserts the output matches `changelog_after.rst`.
The fixtures double as human-readable demos — read alongside the PR
description to see how the system handles patch / minor / major bumps
and cross-fragment section merges.

## Layout

| Demo | Fragments | Bump | Resulting version |
|---|---|---|---|
| `01_patch_bump/` | 2 × `.rst` | patch | `1.2.3 → 1.2.4` |
| `02_minor_bump/` | 1 × `.rst` + 2 × `.minor.rst` | minor | `1.2.3 → 1.3.0` |
| `03_major_bump/` | 1 × `.rst` + 1 × `.minor.rst` + 1 × `.major.rst` | major | `1.2.3 → 2.0.0` |

Each demo includes a `changelog_before.rst` (initial state) and a
`changelog_after.rst` (expected post-compile state). The bump tier is the
**max** of every fragment's filename suffix in the batch.

## Run the compiler against a demo

```bash
./isaaclab.sh -p tools/changelog/cli.py compile --package isaaclab \
    --fragments-dir tools/changelog/test/integration/02_minor_bump/fragments \
    --dry-run
```

`--dry-run` prevents the compile from consuming (deleting) the fixture
fragments. The output should match `02_minor_bump/changelog_after.rst`
modulo today's date.
