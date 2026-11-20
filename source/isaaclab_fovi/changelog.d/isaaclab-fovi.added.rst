* Introduced ``isaaclab_fovi`` and the ``fovi`` installation extra, providing
  :class:`~isaaclab_fovi.sensors.FoveatedCamera`, a foveated camera sensor for the Newton (Warp and OptiX),
  Isaac RTX and OVRTX renderers whose gaze moves inside a physical camera's field of view.
* Added active-gaze manager terms in :mod:`isaaclab_fovi.mdp`: a gaze action with a fixed or tracked camera mount,
  a gaze observation, and teacher gaze labels.
* Added modal-label pooling for segmentation outputs, which keeps their native dtype (``pool_int_mode="modal"``,
  the only accepted value); ``pool_mode`` pools the other outputs and accepts only fovi's modes.
