Fixed
^^^^^

* Fixed ``--video`` recording crashing with ``TypeError: must be real number, not NoneType``
  on fresh installs by bounding the ``moviepy`` dependency to ``>=2.0``. Installs that allow
  prereleases (as the documented Isaac Sim pip install does) resolved the unbounded dependency
  to the broken ``2.0.0.dev2`` build, whose ``write_videofile`` does not fall back to the clip
  fps, so every recorded video failed to encode and training runs exited with an error.
