# Coding Style Examples

## Contents

- Public API change
- Package export change
- Config class with simulator-dependent implementation

## Public API Change

Input: adding a public method to an Isaac Lab class.

Expected workflow:

1. Check naming against `AGENTS.md`.
2. Use specific type hints and Google-style docstrings.
3. Add SI units in public docstrings for physical quantities.
4. Preserve deprecation policy for renamed or removed APIs.
5. Run `uv run isaaclab -f` and focused tests.

## Package Export Change

Input: adding a public symbol to a package `__init__.py`.

Expected workflow:

1. Follow the lazy export pattern from the contribution guide.
2. Update the adjacent `.pyi` stub with explicit public exports.
3. Use relative imports for local submodules.
4. Confirm package import remains lightweight before simulator startup.

## Config Class With Simulator-Dependent Implementation

Input: a config class needs to refer to an implementation that imports simulator runtime modules.

Expected workflow:

1. Avoid eager runtime imports in the config module.
2. Use a resolvable string or `TYPE_CHECKING` guard as documented.
3. Keep the config constructable before `SimulationApp` launches.
4. Add tests or smoke checks for the resolved runtime path.
