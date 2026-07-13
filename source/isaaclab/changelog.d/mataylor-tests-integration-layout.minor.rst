Changed
^^^^^^^

* Moved installation CI tests from ``source/isaaclab/test/install_ci/`` to
  ``tests/integration/install_ci/`` to clearly separate integration and
  installation tests from unit tests. Updated :mod:`tools.run_install_ci`,
  CI actions, and the Dockerfile entrypoint to reference the new location.
  The ``tests/integration/install_ci/`` suite retains its own ``pytest.ini``
  and ``conftest.py`` and is unchanged in behaviour.
