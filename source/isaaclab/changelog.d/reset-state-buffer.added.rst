* Added :class:`~isaaclab.envs.reset.ResetStateBuffer` for owning fixed reset-state
  tensor snapshots independently of their preparation, sampling, and restoration.
  Construction detached and copied the supplied tensors; full-state and selected-row
  accessors returned copies.
