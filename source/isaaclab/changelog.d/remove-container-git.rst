Removed
^^^^^^^

* Removed ``git`` and Git LFS from the base and kit-less container images. CI materialises its
  ``lfs: true`` checkout on the runner before bind-mounting the tree in, and the ``git+https://``
  dependencies still resolve because the kit-less builder stage keeps ``git``. Anything that shells
  out to ``git`` from inside a container now fails, including
  :func:`~isaaclab.utils.assets.retrieve_git_asset_path` for remote git asset sources and the
  benchmark version recorder. Install ``git`` in a derived image, or resolve those paths on the host
  and mount the result.
