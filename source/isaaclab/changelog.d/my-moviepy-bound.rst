Fixed
^^^^^

* Fixed ``--video`` recording crashing with ``TypeError: must be real number, not NoneType``
  on fresh installs by bounding ``moviepy`` to ``>=1.0.3,<2.0.0.dev0``. The unbounded dependency
  let prerelease-allowing installs (as the documented Isaac Sim pip install performs) resolve
  the broken ``2.0.0.dev2`` build, whose ``write_videofile`` does not fall back to the clip fps;
  stable 2.x cannot be used because it caps ``pillow`` below the version Isaac Lab requires.
