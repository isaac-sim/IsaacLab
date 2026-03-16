Coverage-Based Test Selection
=============================

The CI uses a coverage-based test selection system to run only the tests affected by
your changes, rather than the full test suite on every PR.


How It Works
------------

The system has three components:

1. **Coverage mapping** — A nightly workflow (``coverage-map.yml``) runs every test file
   with ``coverage run`` and records which source files each test touches. The resulting
   JSON mapping is committed to the ``ci/coverage-map`` branch as
   ``tools/test-dependency-map.json``.

2. **Test selection** — On every PR, a ``select-tests`` job runs
   ``tools/select_tests.py`` to diff the PR against the base branch, look up which tests
   cover the changed files, and assign them to the appropriate CI jobs (``test-physx``,
   ``test-newton``, ``test-general``, etc.).

3. **Job filtering** — Each CI test job is skipped entirely if it has no selected tests,
   or runs with a filtered ``include-files`` list. When the mapping is missing or stale,
   the system falls back to running all tests.


Fallback Behavior
-----------------

The system falls back to running **all** tests when any of the following are true:

* A changed file is not present in the mapping (e.g. a newly added file).
* The mapping is stale (older than 7 days).
* Non-Python files are changed (YAML, RST, Dockerfiles, etc.).
* CI infrastructure files are changed (``.github/``, ``docker/``, ``tools/conftest.py``).
* The mapping file is empty or missing.

This ensures that the selective system never silently skips tests that should run.


Testing Locally
---------------

Unit tests
^^^^^^^^^^

The selection logic has its own test suite that runs without GPU or simulation:

.. code-block:: bash

   ./isaaclab.sh -p -m pytest tools/test_select_tests.py -v \
       --override-ini="confcutdir=tools" --noconftest

Dry-run against your branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a dummy mapping and run the CLI to see what would be selected:

.. code-block:: bash

   # Create a minimal mapping
   cat > /tmp/test-map.json << 'EOF'
   {
     "metadata": {
       "generated_at": "2026-03-16T00:00:00+00:00",
       "commit": "abc123",
       "test_file_count": 5,
       "source_file_count": 10
     },
     "source_to_tests": {
       "source/isaaclab/isaaclab/utils/math.py": [
         "source/isaaclab/test/utils/test_math.py"
       ]
     }
   }
   EOF

   # Dry-run against main (JSON to stdout, rationale to stderr)
   python tools/select_tests.py \
       --base-branch origin/main \
       --mapping /tmp/test-map.json \
       --dry-run

Fallback behavior
^^^^^^^^^^^^^^^^^

Verify that an empty mapping triggers a full test run:

.. code-block:: bash

   echo '{}' > /tmp/empty-map.json
   python tools/select_tests.py \
       --base-branch origin/main \
       --mapping /tmp/empty-map.json \
       --dry-run

Full-path matching in conftest
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``conftest.py`` include-files filter supports full paths to disambiguate test files
with the same basename across packages (e.g. ``test_articulation.py`` in both
``isaaclab_physx`` and ``isaaclab_newton``):

.. code-block:: bash

   ./isaaclab.sh -p -c "
   from tools.conftest import _matches_include_files

   # Full path matches only the intended package
   assert _matches_include_files(
       'source/isaaclab_physx/test/assets/test_articulation.py',
       'test_articulation.py',
       {'source/isaaclab_physx/test/assets/test_articulation.py'})

   # Does NOT match a different package
   assert not _matches_include_files(
       'source/isaaclab_newton/test/assets/test_articulation.py',
       'test_articulation.py',
       {'source/isaaclab_physx/test/assets/test_articulation.py'})

   # Basename fallback still works
   assert _matches_include_files(
       'source/isaaclab_physx/test/assets/test_articulation.py',
       'test_articulation.py',
       {'test_articulation.py'})

   print('All assertions passed')
   "


Regenerating the Coverage Mapping
---------------------------------

The nightly workflow handles this automatically, but you can also build the mapping on a
local GPU workstation:

.. code-block:: bash

   ./isaaclab.sh -p -m pip install coverage
   ./isaaclab.sh -p tools/collect_coverage_map.py --workers 4 --timeout 2000

The mapping is written to ``tools/test-dependency-map.json``. Increase ``--workers`` if
your machine has sufficient GPU memory — each worker spawns a separate simulation instance.

To push the mapping so CI can use it:

.. code-block:: bash

   git checkout -b ci/coverage-map 2>/dev/null || git checkout ci/coverage-map
   git add tools/test-dependency-map.json
   git commit -m "Update test dependency mapping"
   git push origin ci/coverage-map


CI Integration
--------------

The only way to test the full ``build.yaml`` wiring end-to-end is to push your branch and
open a PR. The ``select-tests`` job runs on ``ubuntu-latest`` and its outputs are visible
in the job logs.

If the ``ci/coverage-map`` branch does not exist yet, the system falls back to
``run-all=true`` (the existing behavior), so it is always safe to merge.
