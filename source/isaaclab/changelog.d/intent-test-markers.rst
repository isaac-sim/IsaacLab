Added
^^^^^

* Added path-derived pytest *intent* markers (``unit``, ``integration``, ``rendering``,
  ``training``, ``performance``, ``benchmark``) that are auto-applied to every test during
  collection by a repository-root ``conftest.py``, so tests can be filtered by kind without
  hand-annotating each file (e.g. ``pytest -m rendering`` or ``pytest -m "unit and not performance"``).
