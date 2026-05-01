# Changelog Examples

Worked examples of the fragment-based changelog system. Each demo is a
self-contained directory with the input fragments and the resulting
`CHANGELOG.rst`, exercised end-to-end by
`tools/changelog/tests/cases/test_integration.py` so it can't drift
from real compiler behavior.

## Layout

```
examples/changelog/
├── README.md                       ← this file
├── 01_patch_bump/                  ← multiple .rst → patch bump
│   ├── fragments/{8001.rst, 8002.rst}
│   ├── changelog_before.rst        starting state (1.2.3)
│   └── changelog_after.rst         compiled state (1.2.4)
├── 02_minor_bump/                  ← mix of .rst + .minor.rst → minor bump
│   ├── fragments/{8003.rst, 8004.minor.rst, 8005.minor.rst}
│   ├── changelog_before.rst        (1.2.3)
│   └── changelog_after.rst         (1.3.0)
└── 03_major_bump/                  ← mix incl. .major.rst → major bump
    ├── fragments/{8006.rst, 8007.minor.rst, 8008.major.rst}
    ├── changelog_before.rst        (1.2.3)
    └── changelog_after.rst         (2.0.0)
```

Each `fragments/` directory contains **multiple** files so the demo also
illustrates the compiler's cross-fragment merge behavior — bullets that
share a section heading (e.g. two `Added` sections from different PRs)
collapse into one section in the compiled output.

## Filename convention

Each fragment's name encodes the **PR number** and the **bump type**:

| Filename | Bump |
|---|---|
| `<pr-number>.rst` | patch (default) |
| `<pr-number>.minor.rst` | minor (new APIs, no breakage) |
| `<pr-number>.major.rst` | major (breaking change / removal) |
| `<pr-number>.skip` | no entry, no bump |

Within a batch the **highest** bump wins for the package
(`major > minor > patch`). The bump tier is the contributor's
declaration; there is no per-tier override at compile time. The
maintainer can pin the resulting version directly with
`compile --package <name> --version X.Y.Z`, which skips the
inference entirely (single-package only — each managed package has
its own version trajectory).

## Run the compiler against any demo

The demos are the easiest way to see what `compile` does without
touching any real package's `changelog.d/`. Always run with `--dry-run`
when pointing at an example — without it, the compile would consume
(delete) the demo's fragments.

```bash
# Bump inferred from filename suffixes (here → minor).
./isaaclab.sh -p tools/changelog/cli.py compile --package isaaclab \
    --fragments-dir examples/changelog/02_minor_bump/fragments \
    --dry-run
```

The dry-run prints the new version block to stdout and reports the
intended `extension.toml` bump, without modifying any file. Compare the
output to the demo's `changelog_after.rst` (modulo today's date) to see
exactly what the compiler would produce.

## Other CLI invocations

```bash
# Real release-time compile across all managed packages — bumps each
# package's extension.toml and CHANGELOG.rst from its own changelog.d/,
# then deletes consumed fragments. Maintainer command.
./isaaclab.sh -p tools/changelog/cli.py compile --all

# Pin one package to a specific version (skips the per-fragment bump
# inference). --version requires --package because each managed package
# has its own version trajectory.
./isaaclab.sh -p tools/changelog/cli.py compile --package isaaclab --version 4.7.0

# PR gate (CI also runs this with --pr ${{ github.event.number }}):
./isaaclab.sh -p tools/changelog/cli.py check develop
```

Run `./isaaclab.sh -p tools/changelog/cli.py compile --help` (or
`check --help`) to see the full flag reference. The module docstring at
the top of `tools/changelog/cli.py` carries the same usage walkthrough.

## What the compiler does, step by step

1. **Globs** `*.rst` in `--fragments-dir` (non-recursive, `.gitkeep` ignored).
   Files that don't match a recognised pattern are skipped with a warning.
2. **Sorts** fragments by **merge time** — the timestamp of the commit that
   introduced each fragment to develop's first-parent history. Older PRs
   appear first within each section.
3. **Parses** each fragment into `{section: [bullets]}`. A section is a
   non-empty line followed by a `^^^^` underline at least as long.
4. **Resolves the bump** as the max of every fragment's filename suffix
   (`patch`/`minor`/`major`).
5. **Merges** sections across fragments — bullets that share a heading
   concatenate into one section.
6. **Emits** a single RST block with `X.Y.Z (YYYY-MM-DD)` heading and
   sections in canonical order: `Added`, `Changed`, `Deprecated`,
   `Removed`, `Fixed` (anything else after).
7. **Prepends** the block to `source/<pkg>/docs/CHANGELOG.rst` and
   updates `version = "X.Y.Z"` in `source/<pkg>/config/extension.toml`.
8. **Deletes** the consumed `*.rst` and any `*.skip` files in the
   fragments directory (`.gitkeep` is preserved).
