Added
^^^^^

* Added pytest *level* markers (``unit`` and ``integration``) and a ``rendering`` *flavor* marker
  registered in the repository-root ``pyproject.toml`` and applied per file via a module-level
  ``pytestmark``, so tests can be filtered by kind (e.g. ``pytest -m unit``, ``pytest -m "not unit"``,
  or ``pytest -m "integration and not rendering"``). The repository-root ``conftest.py`` records
  each test's markers into the JUnit XML report for CI to categorize uploaded results.
