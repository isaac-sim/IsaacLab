* Published model frames on the simulation device through per-view ``FrameBuffer`` objects and per-camera
  ``FrameReader`` objects. Cameras bind to named views and skip copies when frames have not changed.
* Made ``WorldModel`` a pure interface. Implementations are constructed with ``cfg``, ``num_envs`` and ``device``,
  receive device tensors and return one device frame tensor per view. Reset receives per-view image tensors.
