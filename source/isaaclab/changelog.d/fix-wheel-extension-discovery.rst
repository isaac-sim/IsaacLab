Fixed
^^^^^

* Fixed the pip wheel build so that extensions promoted to top-level packages
  (e.g. ``isaaclab_assets``, ``isaaclab_tasks``, ``isaaclab_rl``) keep their
  ``config/extension.toml`` under ``isaaclab/source/`` where the bundled Kit
  experience files search for them. Without this, launching a Kit app from the
  wheel failed during dependency resolution with
  ``isaaclab_assets ... (none found)`` before the simulator started.
