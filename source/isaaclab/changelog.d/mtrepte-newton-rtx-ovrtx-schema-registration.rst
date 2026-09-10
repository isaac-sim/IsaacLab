Fixed
^^^^^

* Fixed benign ``[omni.rtx] FindAppliedAPIPrimDefinition(...) returned nothing`` /
  ``Could not find UsdPrimDefinition for 'OmniRtx...API'`` error logs printed the first time
  the ``newton_rtx`` visualizer opens its render product with ``physics=ovphysx`` by registering
  the OVRTX schemas before physics initializes USD.
