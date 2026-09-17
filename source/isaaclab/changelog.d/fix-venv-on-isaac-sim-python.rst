Fixed
^^^^^

* Fixed ``isaaclab.sh`` and the ``isaaclab`` CLI rejecting a virtual environment that was created on
  a downloaded Isaac Sim package's own Python. Such an environment reuses that interpreter, so Kit's
  extension modules load exactly as they do under the bundled Python; only environments supplying
  their own interpreter, such as conda, are still refused. Create one with
  ``uv venv --python _isaac_sim/kit/python/bin/python3``.
