Fixed
^^^^^

* Fixed the train and play startup summary reporting ``physics=physx`` and ``renderer=rtx`` by the
  name the command line asked for. Both name a backend family that is resolved at launch, so the
  summary now names the backend the run will actually use -- ``default (ovrtx)`` for a kitless
  ``physics=ovphysx renderer=rtx`` run and ``default (isaacsim_rtx)`` when the run needs Kit -- and
  lists the selector itself in the presets row.
* Fixed the summary of a ``--video`` training run reporting no visualizer, which happened because
  the recording visualizer was injected after the summary was printed.
