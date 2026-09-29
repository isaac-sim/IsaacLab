Changed
^^^^^^^

* Selected recording sources from composed visualizer configs instead of resolving CLI viewer
  names again. Declared the recorder and a headless Kit viewer before launch when neither was specified;
  explicit ``visualizer=none`` required a scene-camera recorder.

Fixed
^^^^^

* Preserved physics, renderer, and visualizer selector aliases when filtering external task callback arguments.
