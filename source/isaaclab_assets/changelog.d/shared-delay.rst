Changed
^^^^^^^

* Migrated Spot actuator command latency to explicit ``DelayCfg`` compositions around the hip and knee controllers.
  Controller settings now belonged to each wrapper's ``term``; delay bounds used ``min_lag`` and ``max_lag``.
