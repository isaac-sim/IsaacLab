Changed
^^^^^^^

* Changed :meth:`~isaaclab.utils.buffers.CircularBuffer.append` to apply the first-push
  backfill branchlessly with :func:`torch.where`, removing a host-synchronizing ``nonzero``
  and a device-scalar branch per append while resets are pending.
