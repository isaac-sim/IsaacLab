Added
^^^^^

* Added Newton cable rendering to :class:`~isaaclab_ov.renderers.OVRTXRenderer`. Cable curve points
  are computed on device from the Newton segment bodies and written zero-copy through an OVRTX array
  binding each frame, so cables follow their simulated pose instead of drawing at their spawn pose
  and never moving.
* Added the same cable binding to the ovstage render path. The endpoint kernel still runs on device;
  the handover is host-side because ovstage 0.1.0 accepts the ``points`` column's dtype override
  only on numpy arrays, not on DLPack producers. The legacy path remains zero-copy.
