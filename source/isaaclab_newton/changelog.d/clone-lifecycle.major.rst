Changed
^^^^^^^

* **Breaking:** Removed implicit USD-stage import when Newton started without a builder.
  Use ``InteractiveScene`` to construct and replicate configured USD assets before initialization.
  Native tools can continue supplying a builder
  through ``NewtonManager.set_builder(builder)`` without declaring a clone plan.

Fixed
^^^^^

* Prevented globally declared native deformables from being imported twice through clone plans.
