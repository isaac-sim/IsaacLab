# Coding Style Examples

## Contents

- Stateless computation
- Public API change
- Package export change
- Config class with simulator-dependent implementation

## Stateless Computation

Input: adding a computation with no persistent state or required class interface.

Expected workflow:

1. Apply the guide's functional-design and helper-extraction criteria.
2. Check whether an existing function already owns the operation.
3. Follow the code skeleton for an operation that warrants its own function; keep trivial expressions inline.
4. Use the test-audit gate to decide whether existing coverage needs extension.

## Public API Change

Input: adding a public function or a method on an existing Isaac Lab class.

Expected workflow:

1. Check naming against the contribution guide.
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
2. Use a resolvable string for the runtime reference and a `TYPE_CHECKING` guard for its annotation, as documented.
3. Keep the config constructable before `SimulationApp` launches.
4. Run existing coverage for the resolved runtime path. Apply the test-audit authoring gate before adding coverage.
