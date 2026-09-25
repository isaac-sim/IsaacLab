Changed
^^^^^^^

* **Breaking:** Restricted ``ManagerBasedEnv.reset`` / ``reset_to`` and reset event selectors to one-dimensional
  ``torch.int32`` / ``torch.int64`` tensors on the environment device, positive-step slices, or ``None``.
  Python sequences and tensors on a different device now raise before resetting state instead of
  relying on implicit indexing conversions. Callers using explicit lists must construct their index
  tensor on the environment device once and reuse it.
* Made reset event callbacks consistently receive device index tensors, including for ``None`` and
  slice selectors. Resolved those selectors through the scene's existing index buffer without copying
  index data or transferring it between devices.

Fixed
^^^^^

* Resolved int64 environment indices to the native CPU int32 representation when writing PhysX rigid-body
  material properties.
