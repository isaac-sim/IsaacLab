Removed
^^^^^^^

* Removed MoviePy and its bundled FFmpeg backend from the default Docker
  installation. Install them explicitly with
  ``uv pip install "moviepy>=1.0.3,<2.0.0.dev0"`` before using ``--video``.
