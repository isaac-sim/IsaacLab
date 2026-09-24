Changed
^^^^^^^

* **Breaking:** Removed implicit USD-stage import when Newton started without a builder.
  Standalone USD workflows must declare and replicate a clone plan before initialization;
  ``InteractiveScene`` already handles this. Native tools can continue supplying a builder
  through ``NewtonManager.set_builder(builder)`` without declaring a clone plan.

Fixed
^^^^^

* Prevented globally declared native deformables from being imported twice through clone plans.
