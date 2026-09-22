Changed
^^^^^^^

* **Breaking:** Updated the optional OV dependencies to ``ovrtx==0.5.0.377615``,
  ``ovstage==0.2.0.377349``, ``ovphysx==0.6.3``, and ``omniverseclient==2.74.0``. Upgrade
  them together with ``uv sync --inexact --extra ov``.
* **Breaking:** OvPhysX CPU-backed property writes require CPU-resident ``values``,
  ``indices``, and ``mask`` arrays.
