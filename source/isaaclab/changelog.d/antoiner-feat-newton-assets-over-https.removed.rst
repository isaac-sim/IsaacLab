* **Breaking:** Removed :func:`~isaaclab.utils.assets.retrieve_git_asset_path`,
  ``isaaclab.utils.assets.NEWTON_ASSET_REPO_URL``, and ``isaaclab.utils.assets.GIT_ASSET_CACHE_DIR``. Pass
  ``f"{NEWTON_ASSET_DIR}/<path>"`` to :func:`~isaaclab.utils.assets.retrieve_file_path` or to a spawner's
  ``usd_path`` instead: :data:`~isaaclab.utils.assets.NEWTON_ASSET_DIR` now points at the Newton asset
  repository over HTTPS, which downloads only the files an asset references instead of cloning the repository.
