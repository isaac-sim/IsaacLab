Fixed
^^^^^

* Fixed standalone ``isaaclab_rl`` installs downgrading ``pillow`` below the
  version prebundled by Isaac Sim: ``moviepy`` 2.x caps ``pillow<12.0``, and
  on aarch64 the forced downgrade deleted the prebundled copy that
  ``omni.kit.pip_archive`` shares via per-file symlinks, breaking extension
  startup. Added a ``pillow>=12.1.1`` floor so pip resolves ``moviepy`` 1.0.3
  instead of touching pillow.
