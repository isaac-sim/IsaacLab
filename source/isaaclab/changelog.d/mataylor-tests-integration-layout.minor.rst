Changed
^^^^^^^

* Moved installation CI tests from ``source/isaaclab/test/install_ci/`` to
  ``tests/integration/install_ci/`` to clearly separate integration and
  installation tests from unit tests. Updated :mod:`tools.run_install_ci`,
  CI actions, and the Dockerfile entrypoint to reference the new location.
  The ``tests/integration/install_ci/`` suite retains its own ``pytest.ini``
  and ``conftest.py`` and is unchanged in behaviour.

* Moved micro-benchmark tests from ``source/isaaclab/test/benchmark/`` to
  ``tests/integration/benchmark/`` so they live alongside other
  cross-package integration tests rather than alongside unit tests.

* Moved the unit-test import policy test from ``source/isaaclab/test/cli/``
  to ``tests/integration/`` and updated :mod:`tools.conftest` to scan
  ``tests/integration/`` so future integration tests placed there are
  picked up automatically.

* Declared ``scipy`` as an explicit dependency in the root ``pyproject.toml``.
  It was already used in production code and unit tests but was undeclared,
  making it a silent transitive dependency.

* Added a testing guideline to ``AGENTS.md`` requiring unit tests to only
  import packages declared in the root ``pyproject.toml``; tests that need
  unlisted packages must either declare the dependency or be reclassified as
  integration tests.
