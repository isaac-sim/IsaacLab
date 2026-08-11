# IsaacLab Development Guidelines
- Make the smallest change that solves the requested problem.
- Preserve existing behavior unless the task explicitly changes it.
- Read surrounding code, callers, tests, and documentation before changing an interface.
- Do not add dependencies unless necessary. Prefer existing project dependencies and the standard library.
- Do not commit generated plans, scratch files, or agent artifacts.
- Follow a more-specific `AGENTS.md` in the directory being changed.
- Use the repository's current SPDX header template for new source files; do not change existing file headers.
- Follow the existing style and abstractions in the affected package.
- Use modern Python type hints, including `X | None` instead of `Optional[X]`.
- Use `snake_case` for methods, functions, and CLI arguments.
- Keep related public symbols discoverable through consistent prefixes.
- Use concrete types for public interfaces where practical.
- Use Google-style docstrings for public APIs.
- Document SI units for public physical quantities in docstrings using inline `[unit]` notation (e.g. `Particle positions [m], shape [N, 3]`); use `[m or rad, depending on joint type]` where applicable, and skip non-physical fields (indices, counts, flags).
- Keep comments brief and explain intent, constraints, or edge cases rather than restating code.
- Do not remove or rename a public API without a prior deprecation and migration path.
- Update public documentation when adding or changing public APIs.
- Verify documented technical claims against the current code and primary sources before relying on them.
- Use the uv-managed environment for routine commands.
- Use `uv run python` for Python scripts and tests.
- Use `uv run isaaclab` for Isaac Lab CLI commands.
- Use `./isaaclab.sh` only for installer workflows that require it.
- Do not define Warp kernels in `python -c`; write a temporary Python file instead so Warp can inspect the source.

## Testing and validation

- Run the narrowest relevant test first.
- Before reporting that a test cannot run because an optional dependency is missing, identify its project extra and retry with `uv run --extra <extra> ...`.
- Do not treat an optional dependency missing from the base environment as a blocker when its project extra is available.
- Run `uv run isaaclab -f` before committing.
- For a regression test, verify that it fails without the fix and passes with it.
- Find and extend the closest existing test before creating a new test file or test case.
- Add a test only when it covers a distinct behavior, regression, boundary, or failure mode that existing tests do not cover clearly.
- Test observable behavior and public contracts, not implementation details.
- Use hard-coded values only when they are the intended contract or a small, independently verified example; otherwise derive the expected result from a separate, simple reference calculation.
- Keep tests focused and remove or consolidate redundant coverage instead of growing overlapping test suites.
- Do not add debug output to production Warp kernels. Use temporary standalone reproductions and remove debug output before committing.

## Changelog and release metadata

- Do not edit `CHANGELOG.rst` or `config/extension.toml` directly.
- Add one changelog fragment for each changed source package when the change is user-visible.
- Use `.skip` fragments for changes that do not require a release note.
- Write changelog entries in past tense and include migration guidance for deprecated, changed, or removed behavior.
- Mark breaking changes clearly and provide migration guidance.

## Commits and branches

- Work on a feature branch; do not commit directly to `develop`, `main`, or any `release/*` branch.
- Keep commits focused and atomic.
- Use an imperative, capitalized commit subject with no trailing period.
- Inspect staged changes before committing.
- Do not add AI co-author or attribution lines.
- Prefer follow-up commits over amending commits while addressing review feedback.

## Repository skills

- Keep repository-owned skills in `skills/`; do not duplicate their contents in tool-specific discovery directories.
- Validate skill changes with `uv run --no-project python tools/skills/cli.py check`.
- Keep skills concise and point to maintained documentation and source examples.
