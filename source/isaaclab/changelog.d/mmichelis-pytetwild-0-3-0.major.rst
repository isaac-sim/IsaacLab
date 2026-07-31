Changed
^^^^^^^

* **Breaking:** Changed automatic volume tetrahedralization to require the
  ``tetrahedralization`` optional dependencies. Install them with
  ``uv sync --inexact --extra tetrahedralization``, or with
  ``pip install "isaaclab[tetrahedralization]"`` from a wheel, or
  ``./isaaclab.sh -i tetrahedralization`` with the legacy installer. The optional
  ``pytetwild`` dependency now uses the 0.3 release series, adding Linux aarch64
  support alongside Linux x86_64 and Windows amd64.
