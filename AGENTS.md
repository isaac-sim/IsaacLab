# IsaacLab Development Guidelines

## Contribution guide

Read the relevant sections of [the contribution guide](docs/source/refs/contributing.rst) before starting:

- [Coding Style](docs/source/refs/contributing.rst#coding-style) for implementation, refactoring, and review.
- [Unit Testing](docs/source/refs/contributing.rst#unit-testing) for test changes and validation;
  use the [test-audit skill](skills/developer/test-audit/SKILL.md) when adding, changing, reviewing, or pruning tests.
- [Contributing Documentation](docs/source/refs/contributing.rst#contributing-documentation) for documentation changes.
- [Maintaining package changelogs and versions](docs/source/refs/contributing.rst#maintaining-package-changelogs-and-versions)
  for source package changes.
- [Tools](docs/source/refs/contributing.rst#tools) for formatting and lint checks.

The guide owns shared contribution rules. Update them there instead of copying them into this file or skills.

## Agent workflow

- Follow a more-specific `AGENTS.md` in the directory being changed.
- Preserve unrelated workspace changes and do not commit generated plans, scratch files, or agent artifacts.
- Use the repository's current SPDX header template for new source files; do not change existing file headers.
- Follow the existing style and abstractions in the affected package.
- Use the uv-managed environment for routine commands and `uv run python` for Python scripts.
  Use `./isaaclab.sh` only for installer workflows that require it.
- Run the guide's formatting and lint checks before committing.
- Do not define Warp kernels in `python -c`; write a temporary Python file instead so Warp can inspect the source.
- Do not add debug output to production Warp kernels. Use temporary standalone reproductions and remove debug output before committing.

## Commits and branches

- Work on a feature branch; do not commit directly to `main`.
- Keep commits focused and atomic.
- Use an imperative, capitalized commit subject with no trailing period.
- Inspect staged changes before committing.
- Do not add AI co-author or attribution lines.
- Prefer follow-up commits over amending commits while addressing review feedback.

## Repository skills

- Keep repository-owned skills in `skills/`; do not duplicate their contents in tool-specific discovery directories.
- Validate skill changes with `uv run --no-project python tools/skills/cli.py check`.
- Keep skills concise and point to maintained documentation and source examples.
